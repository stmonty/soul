//! Node-side execution shell. Persist identity before effects; recover by inspecting
//! the same container rather than creating another attempt. One daemon per node.
use crate::execution::{Command, Operation, Phase, Report};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::{path::Path, process::Command as Process};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContainerState {
    Created,
    Running,
    Exited(i32),
}

pub trait Runtime {
    fn inspect(&mut self, name: &str) -> Result<Option<ContainerState>>;
    fn prepare(&mut self, name: &str, operation: &Operation) -> Result<()>;
    fn start(&mut self, name: &str) -> Result<()>;
    fn stop(&mut self, name: &str) -> Result<()>;
}

pub struct Docker {
    pub controller_url: String,
    pub cpu_test: bool,
    /// Operator opt-in: only for trusted training workloads on dedicated nodes.
    pub host_network: bool,
}
impl Docker {
    fn owner_label(name: &str) -> &'static str {
        if name.starts_with("wizard-") {
            "wizard.owner"
        } else {
            "soul.owner"
        }
    }

    fn run(args: &[String]) -> Result<String> {
        let output = Process::new("docker")
            .args(args)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().into())
    }
}
impl Runtime for Docker {
    fn inspect(&mut self, name: &str) -> Result<Option<ContainerState>> {
        // A successful empty listing proves absence. A Docker connection error does not.
        let found = Self::run(&[
            "container".into(),
            "ls".into(),
            "--all".into(),
            "--filter".into(),
            format!("name=^/{name}$"),
            "--format".into(),
            "{{.Names}}".into(),
        ])?;
        if found.is_empty() {
            return Ok(None);
        }
        let json = Self::run(&[
            "inspect".into(),
            "--format".into(),
            "{{json .}}".into(),
            name.into(),
        ])?;
        let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        let labels = &value["Config"]["Labels"];
        if labels[Self::owner_label(name)].as_str() != Some(name) {
            return Err("refusing to manage a container without the expected owner label".into());
        }
        match value["State"]["Status"].as_str() {
            Some("created") => Ok(Some(ContainerState::Created)),
            Some("running") => Ok(Some(ContainerState::Running)),
            Some("exited" | "dead") => Ok(Some(ContainerState::Exited(
                value["State"]["ExitCode"].as_i64().unwrap_or(255) as i32,
            ))),
            _ => Err("container state is ambiguous; retaining allocation".into()),
        }
    }
    fn prepare(&mut self, name: &str, operation: &Operation) -> Result<()> {
        let Operation::Prepare {
            image,
            command,
            gpus,
            rank,
            nodes,
            rendezvous,
            runtime,
            restore,
        } = operation
        else {
            return Err("not a prepare command".into());
        };
        if nodes.len() > 1 && (!self.host_network || rendezvous.is_none()) {
            return Err(
                "distributed Docker jobs require --host-network and a rendezvous endpoint".into(),
            );
        }
        let mut args = vec![
            "create".into(),
            "--pull=never".into(),
            "--name".into(),
            name.into(),
            "--label".into(),
            format!("{}={name}", Self::owner_label(name)),
            "--cap-drop".into(),
            "ALL".into(),
            "--security-opt".into(),
            "no-new-privileges".into(),
            "--pids-limit".into(),
            "1024".into(),
            "--env".into(),
            format!("SOUL_NODE_RANK={rank}"),
            "--env".into(),
            format!("SOUL_NODE_COUNT={}", nodes.len()),
        ];
        args.extend(["--env".into(), format!("SOUL_GPUS_PER_NODE={}", gpus.len())]);
        if let Some(access) = runtime {
            if !self.host_network {
                return Err("checkpoint coordinator requires --host-network".into());
            }
            args.extend([
                "--env".into(),
                format!("SOUL_RUNTIME_URL={}{}", self.controller_url, access.path),
                "--env".into(),
                format!("SOUL_RUNTIME_TOKEN={}", access.token),
            ]);
        }
        if let Some(checkpoint) = restore {
            args.extend([
                "--env".into(),
                format!("SOUL_CHECKPOINT_REF={}", checkpoint.reference),
                "--env".into(),
                format!("SOUL_CHECKPOINT_PROGRESS={}", checkpoint.progress),
            ]);
        }
        if self.host_network {
            args.extend(["--network".into(), "host".into()]);
        }
        if let Some(rendezvous) = rendezvous {
            args.extend([
                "--env".into(),
                format!("MASTER_ADDR={}", rendezvous.address),
                "--env".into(),
                format!("MASTER_PORT={}", rendezvous.port),
                "--env".into(),
                format!("SOUL_RENDEZVOUS_ID={}", rendezvous.id),
            ]);
        }
        if !self.cpu_test {
            args.extend([
                "--gpus".into(),
                format!(
                    "\"device={}\"",
                    gpus.iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            ]);
        }
        args.push(image.clone());
        args.extend(command.iter().cloned());
        Self::run(&args).map(|_| ())
    }
    fn start(&mut self, name: &str) -> Result<()> {
        Self::run(&["start".into(), name.into()]).map(|_| ())
    }
    fn stop(&mut self, name: &str) -> Result<()> {
        // Workloads must handle SIGTERM and save an application checkpoint during
        // this grace period. This is not transparent process checkpointing.
        Self::run(&["stop".into(), "--time".into(), "30".into(), name.into()]).map(|_| ())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    #[serde(default)]
    container: String,
    phase: Phase,
    exit_code: Option<i32>,
}

pub struct Journal {
    connection: Connection,
    namespace: String,
}
impl Journal {
    pub fn open(path: impl AsRef<Path>, namespace: String) -> Result<Self> {
        let connection = Connection::open(path).map_err(|e| e.to_string())?;
        connection.execute_batch("PRAGMA locking_mode=EXCLUSIVE; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), namespace TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (job TEXT NOT NULL, attempt INTEGER NOT NULL, body TEXT NOT NULL, PRIMARY KEY(job, attempt));")
            .map_err(|e| e.to_string())?;
        connection
            .execute("INSERT OR IGNORE INTO identity VALUES(1, ?1)", [&namespace])
            .map_err(|e| e.to_string())?;
        let saved: String = connection
            .query_row("SELECT namespace FROM identity WHERE id=1", [], |r| {
                r.get(0)
            })
            .map_err(|e| e.to_string())?;
        if saved != namespace {
            return Err("node journal belongs to a different controller or node".into());
        }
        Ok(Self {
            connection,
            namespace,
        })
    }
    pub fn name(&self, job: &str, attempt: u32) -> String {
        format!("soul-{}-{job}-{attempt}", self.namespace)
    }
    fn get(&self, job: &str, attempt: u32) -> Result<Option<Record>> {
        let body: Option<String> = self
            .connection
            .query_row(
                "SELECT body FROM attempts WHERE job=?1 AND attempt=?2",
                rusqlite::params![job, attempt],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        body.map(|body| {
            let mut record: Record = serde_json::from_str(&body).map_err(|e| e.to_string())?;
            // Earlier journals derived the name. A live attempt must keep its original container.
            if record.container.is_empty() {
                record.container = format!("wizard-{}-{job}-{attempt}", self.namespace);
            }
            Ok(record)
        })
        .transpose()
    }
    fn put(&self, job: &str, attempt: u32, record: &Record) -> Result<()> {
        self.connection.execute("INSERT INTO attempts VALUES(?1, ?2, ?3) ON CONFLICT(job, attempt) DO UPDATE SET body=excluded.body",
            rusqlite::params![job, attempt, serde_json::to_string(record).map_err(|e| e.to_string())?]).map_err(|e| e.to_string())?;
        Ok(())
    }
    fn observed(
        &self,
        job: &str,
        attempt: u32,
        mut record: Record,
        state: Option<ContainerState>,
    ) -> Result<Report> {
        (record.phase, record.exit_code) = match state {
            Some(ContainerState::Created) => (Phase::Prepared, None),
            Some(ContainerState::Running) => (Phase::Running, None),
            Some(ContainerState::Exited(code)) => (Phase::Exited, Some(code)),
            None => (Phase::Exited, Some(255)),
        };
        self.put(job, attempt, &record)?;
        Ok(Report {
            job: job.into(),
            attempt,
            phase: record.phase,
            exit_code: record.exit_code,
        })
    }
    pub fn execute(&self, command: &Command, runtime: &mut impl Runtime) -> Result<Report> {
        if !crate::model::valid_id(&command.job) || command.attempt == 0 {
            return Err("invalid command identity".into());
        }
        let record = match self.get(&command.job, command.attempt)? {
            Some(record) if record.phase == Phase::Exited => {
                return Ok(Report {
                    job: command.job.clone(),
                    attempt: command.attempt,
                    phase: record.phase,
                    exit_code: record.exit_code,
                });
            }
            Some(record) => record,
            None => {
                let record = Record {
                    container: self.name(&command.job, command.attempt),
                    phase: Phase::Pending,
                    exit_code: None,
                };
                self.put(&command.job, command.attempt, &record)?;
                record
            }
        };
        let name = record.container.as_str();
        let state = runtime.inspect(name)?;
        match &command.operation {
            Operation::Prepare { .. } if state.is_none() && record.phase == Phase::Pending => {
                runtime.prepare(name, &command.operation)?;
            }
            Operation::Start if state == Some(ContainerState::Created) => runtime.start(name)?,
            Operation::Stop
                if matches!(
                    state,
                    Some(ContainerState::Created | ContainerState::Running)
                ) =>
            {
                if state == Some(ContainerState::Running) {
                    runtime.stop(name)?;
                }
                // A never-started container is also safely stopped; retain its tombstone.
                if state == Some(ContainerState::Created) {
                    return self.observed(
                        &command.job,
                        command.attempt,
                        record,
                        Some(ContainerState::Exited(0)),
                    );
                }
            }
            _ => {}
        }
        let state = runtime.inspect(name)?;
        self.observed(&command.job, command.attempt, record, state)
    }
    pub fn observe(
        &self,
        job: &str,
        attempt: u32,
        runtime: &mut impl Runtime,
    ) -> Result<Option<Report>> {
        let Some(record) = self.get(job, attempt)? else {
            return Ok(None);
        };
        if record.phase == Phase::Exited {
            return Ok(Some(Report {
                job: job.into(),
                attempt,
                phase: record.phase,
                exit_code: record.exit_code,
            }));
        }
        // A pending creation may have failed transiently; allow prepare to retry it.
        if record.phase == Phase::Pending {
            return Ok(None);
        }
        let state = runtime.inspect(&record.container)?;
        Ok(Some(self.observed(job, attempt, record, state)?))
    }
}
