//! Desired-state execution protocol. A reservation is never a claim that code stopped.
use crate::{
    model::*,
    scheduler::{self, Action, Decision},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const HEARTBEAT_TIMEOUT: u64 = 30;
pub const STARTUP_TIMEOUT: u64 = 300;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Pending,
    Prepared,
    Running,
    Exited,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Member {
    pub phase: Phase,
    pub exit_code: Option<i32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Cancel,
    Preempt,
    Resize,
    Failure,
}

/// One controller-reserved TCP endpoint per distributed attempt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rendezvous {
    pub address: String,
    pub port: u16,
    pub id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Execution {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_token: Option<String>,
    #[serde(default)]
    pub rendezvous: Option<Rendezvous>,
    pub attempt: u32,
    pub members: BTreeMap<String, Member>,
    pub running_at: Option<Timestamp>,
    pub stop: Option<StopReason>,
}

impl Execution {
    fn past_prepare_barrier(&self) -> bool {
        self.members
            .values()
            .all(|member| member.phase != Phase::Pending)
    }

    /// Return true only for a new observation from a member of this attempt.
    fn observe_member(&mut self, node: &str, report: &Report) -> Result<bool, String> {
        if self.attempt != report.attempt {
            return Err("stale attempt generation".into());
        }
        let member = self
            .members
            .get(node)
            .ok_or("node does not own this attempt")?;
        if (report.phase == Phase::Exited) != report.exit_code.is_some() {
            return Err("exit code is required only for an exited report".into());
        }
        if member.phase == report.phase && member.exit_code == report.exit_code {
            return Ok(false);
        }
        if member.phase == Phase::Exited || report.phase == Phase::Pending {
            return Err("invalid report transition".into());
        }
        match report.phase {
            Phase::Prepared if member.phase != Phase::Pending => {
                return Err("invalid prepared transition".into());
            }
            Phase::Running if member.phase != Phase::Prepared || !self.past_prepare_barrier() => {
                return Err("job has not passed preparation barrier".into());
            }
            _ => {}
        }
        let member = self.members.get_mut(node).unwrap();
        member.phase.clone_from(&report.phase);
        member.exit_code = report.exit_code;
        Ok(true)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Operation {
    Prepare {
        image: String,
        command: Vec<String>,
        gpus: Vec<u32>,
        rank: usize,
        nodes: Vec<String>,
        #[serde(default)]
        rendezvous: Option<Rendezvous>,
        #[serde(default)]
        runtime: Option<Box<RuntimeAccess>>,
        #[serde(default)]
        restore: Option<crate::resize::Checkpoint>,
    },
    Start,
    Stop,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeAccess {
    pub path: String,
    pub token: String,
}

/// Stable (job, attempt, operation) identity; retries must not duplicate execution.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Command {
    pub job: String,
    pub attempt: u32,
    pub operation: Operation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub job: String,
    pub attempt: u32,
    pub phase: Phase,
    pub exit_code: Option<i32>,
}

/// Reconcile one admission per pool. Stops are durable intent; do not apply the
/// simulator's instantaneous preemption. Replan after termination is confirmed.
pub fn reconcile(cluster: &mut Cluster) -> Vec<Decision> {
    if cluster.mode != Mode::Managed {
        return vec![];
    }
    let timed_out: Vec<_> = cluster
        .jobs
        .values()
        .filter(|job| {
            job.state == JobState::Starting
                && cluster
                    .now
                    .saturating_sub(job.attempts.last().unwrap().started_at)
                    >= STARTUP_TIMEOUT
        })
        .map(|job| job.spec.id.clone())
        .collect();
    for id in timed_out {
        request_stop(cluster, &id, StopReason::Failure).expect("valid starting execution");
    }
    let pools: Vec<_> = cluster.pools.keys().cloned().collect();
    let mut decisions = vec![];
    for pool in pools {
        let Some(mut decision) = scheduler::next(cluster, &pool) else {
            continue;
        };
        match &decision.action {
            Action::Start { placement, preempt } if preempt.is_empty() => {
                if let Err(reason) = reserve_gang(cluster, &decision.job, placement) {
                    decision.reason = reason;
                    decision.action = Action::Wait;
                }
            }
            Action::Start { preempt, .. } => {
                for id in preempt {
                    request_stop(cluster, id, StopReason::Preempt).expect("eligible victim");
                }
            }
            Action::Resize { request, estimate } => {
                crate::resize::record_request(
                    cluster,
                    &decision.job,
                    request.clone(),
                    estimate.clone(),
                )
                .expect("validated growth proposal");
            }
            Action::Wait => {}
        }
        decisions.push(decision);
    }
    decisions
}

fn reserve_gang(cluster: &mut Cluster, id: &str, placement: &[Placement]) -> Result<(), String> {
    let rendezvous = reserve_rendezvous(cluster, id, placement)?;
    let job = cluster.jobs.get_mut(id).unwrap();
    let number = job.reserve_attempt(cluster.now, placement);
    cluster.executions.insert(
        id.to_owned(),
        Execution {
            runtime_token: None,
            rendezvous,
            attempt: number,
            running_at: None,
            stop: None,
            members: placement
                .iter()
                .map(|p| {
                    (
                        p.node.clone(),
                        Member {
                            phase: Phase::Pending,
                            exit_code: None,
                        },
                    )
                })
                .collect(),
        },
    );
    Ok(())
}

pub fn heartbeat(cluster: &mut Cluster, node: &str) -> Result<(), String> {
    if !cluster.nodes.contains_key(node) {
        return Err("unknown node".into());
    }
    cluster.heartbeats.insert(node.into(), cluster.now);
    Ok(())
}

pub fn commands(cluster: &Cluster, node: &str) -> Result<Vec<Command>, String> {
    if !cluster.nodes.contains_key(node) {
        return Err("unknown node".into());
    }
    let mut commands = vec![];
    for (id, execution) in &cluster.executions {
        let Some(member) = execution.members.get(node) else {
            continue;
        };
        if member.phase == Phase::Exited {
            continue;
        }
        let job = &cluster.jobs[id];
        let operation = if execution.stop.is_some() {
            Operation::Stop
        } else if job.spec.resize_policy.is_some() && !cluster.checkpoint_nodes.contains(node) {
            continue;
        } else if member.phase == Phase::Pending {
            let placement = &job.attempts.last().unwrap().placement;
            let rank = placement.iter().position(|p| p.node == node).unwrap();
            Operation::Prepare {
                image: job.spec.image.clone(),
                command: job.spec.command.clone(),
                gpus: placement[rank].gpus.clone(),
                rank,
                runtime: if rank == 0 && job.spec.resize_policy.is_some() {
                    let Some(token) = &execution.runtime_token else {
                        continue;
                    };
                    Some(Box::new(RuntimeAccess {
                        path: format!("/v1/runtime/{id}/{}", execution.attempt),
                        token: token.clone(),
                    }))
                } else {
                    None
                },
                restore: job
                    .attempts
                    .iter()
                    .rev()
                    .nth(1)
                    .and_then(Attempt::checkpoint)
                    .map(|(_, c)| c.clone()),
                nodes: placement.iter().map(|p| p.node.clone()).collect(),
                rendezvous: execution.rendezvous.clone(),
            }
        } else if member.phase == Phase::Prepared && execution.past_prepare_barrier() {
            Operation::Start
        } else {
            continue;
        };
        commands.push(Command {
            job: id.clone(),
            attempt: execution.attempt,
            operation,
        });
    }
    Ok(commands)
}

pub fn request_stop(cluster: &mut Cluster, id: &str, reason: StopReason) -> Result<(), String> {
    if cluster.mode != Mode::Managed {
        return Err("not a managed controller".into());
    }
    let job = cluster.jobs.get_mut(id).ok_or("unknown job")?;
    match job.state {
        JobState::Completed | JobState::Cancelled | JobState::Failed => return Ok(()),
        JobState::Queued => {
            if reason != StopReason::Cancel {
                return Err("queued job cannot be preempted".into());
            }
            job.state = JobState::Cancelled;
            return Ok(());
        }
        _ => {}
    }
    if reason == StopReason::Resize && job.attempts.last().and_then(Attempt::checkpoint).is_none() {
        return Err("resize requires a durable checkpoint acknowledgement".into());
    }
    job.attempts
        .last_mut()
        .unwrap()
        .close_pending_resize(cluster.now, crate::resize::Outcome::Interrupted);
    let execution = cluster.executions.get_mut(id).ok_or("missing execution")?;
    // Explicit cancellation wins over a pending retry/preemption.
    if execution.stop.is_none() || reason == StopReason::Cancel {
        execution.stop = Some(reason);
    }
    job.state = JobState::Stopping;
    Ok(())
}

pub fn report(cluster: &mut Cluster, node: &str, report: Report) -> Result<(), String> {
    if cluster.mode != Mode::Managed {
        return Err("not a managed controller".into());
    }
    let execution = cluster
        .executions
        .get_mut(&report.job)
        .ok_or("unknown execution")?;
    if !execution.observe_member(node, &report)? {
        return Ok(());
    }
    let job = cluster.jobs.get_mut(&report.job).unwrap();
    if report.phase == Phase::Exited {
        job.attempts
            .last_mut()
            .unwrap()
            .close_pending_resize(cluster.now, crate::resize::Outcome::Interrupted);
    }
    if execution.stop.is_none()
        && execution
            .members
            .values()
            .all(|m| matches!(m.phase, Phase::Running | Phase::Exited))
    {
        execution.running_at.get_or_insert(cluster.now);
        job.state = JobState::Running;
    }
    if execution.stop.is_none()
        && execution
            .members
            .values()
            .any(|m| m.phase == Phase::Exited && m.exit_code != Some(0))
    {
        execution.stop = Some(StopReason::Failure);
        job.state = JobState::Stopping;
    }
    if execution.members.values().all(|m| m.phase == Phase::Exited) {
        job.attempts.last_mut().unwrap().release(cluster.now);
        job.state = match execution.stop {
            Some(StopReason::Cancel) => JobState::Cancelled,
            Some(StopReason::Preempt | StopReason::Resize) => JobState::Queued,
            Some(StopReason::Failure) => JobState::Failed,
            None => JobState::Completed,
        };
    }
    Ok(())
}

/// Validate persisted state before policy or execution code can dereference it.
pub fn validate(cluster: &Cluster) -> Result<(), String> {
    let mut occupied = BTreeSet::new();
    for (id, job) in &cluster.jobs {
        job.spec.validate_shape()?;
        crate::resize::validate_history(job, cluster.now)?;
        if id != &job.spec.id
            || !cluster.accounts.contains_key(&job.spec.account)
            || cluster.accounts[&job.spec.account].pool != job.spec.pool
        {
            return Err("invalid persisted job".into());
        }
        let active = job.state.holds_allocation();
        if active && job.attempts.last().is_some_and(|a| a.ended_at.is_some()) {
            return Err("active attempt already ended".into());
        }
        if active && job.attempts.is_empty() {
            return Err("active job has no attempt".into());
        }
        for (index, attempt) in job.attempts.iter().enumerate() {
            if attempt.number as usize != index + 1
                || attempt.started_at > cluster.now
                || attempt
                    .ended_at
                    .is_some_and(|end| end < attempt.started_at || end > cluster.now)
                || (attempt.ended_at.is_none() && (!active || index + 1 != job.attempts.len()))
            {
                return Err("invalid attempt interval".into());
            }
            let mut nodes = BTreeSet::new();
            if !job
                .spec
                .node_counts()
                .any(|n| n as usize == attempt.placement.len())
            {
                return Err("invalid gang size".into());
            }
            for placement in &attempt.placement {
                let node = cluster
                    .nodes
                    .get(&placement.node)
                    .ok_or("unknown placement node")?;
                if node.pool != job.spec.pool
                    || (job.spec.same_gpu_domain
                        && node.select_devices(&placement.gpus, &job.spec).is_none())
                    || !nodes.insert(&placement.node)
                    || placement.gpus.len() != job.spec.gpus_per_node as usize
                {
                    return Err("invalid placement".into());
                }
                let mut gpus = BTreeSet::new();
                for gpu in &placement.gpus {
                    if *gpu >= node.gpus || !gpus.insert(gpu) {
                        return Err("invalid GPU allocation".into());
                    }
                    if active
                        && index + 1 == job.attempts.len()
                        && !occupied.insert((&placement.node, gpu))
                    {
                        return Err("GPU allocated to multiple attempts".into());
                    }
                }
            }
        }
        if cluster.mode == Mode::Managed && active && !cluster.executions.contains_key(id) {
            return Err("active managed job has no execution".into());
        }
    }
    let mut endpoints = BTreeSet::new();
    for (id, execution) in &cluster.executions {
        let job = cluster.jobs.get(id).ok_or("orphan execution")?;
        let attempt = job.attempts.last().ok_or("execution without attempt")?;
        if execution.stop == Some(StopReason::Resize) && attempt.checkpoint().is_none() {
            return Err("resize stop lacks checkpoint".into());
        }
        if execution.attempt != attempt.number
            || execution.members.keys().collect::<BTreeSet<_>>()
                != attempt.placement.iter().map(|p| &p.node).collect()
        {
            return Err("execution does not match allocation".into());
        }
        if let Some(rendezvous) = &execution.rendezvous {
            if !(29400..29500).contains(&rendezvous.port)
                || attempt.placement.len() < 2
                || cluster.nodes[&attempt.placement[0].node].address.as_ref()
                    != Some(&rendezvous.address)
                || rendezvous.id != format!("{id}-{}", attempt.number)
            {
                return Err("invalid rendezvous reservation".into());
            }
            if execution.members.values().any(|m| m.phase != Phase::Exited)
                && !endpoints.insert((&rendezvous.address, rendezvous.port))
            {
                return Err("rendezvous endpoint allocated to multiple attempts".into());
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Assignment {
    pub job: String,
    pub attempt: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Work {
    pub controller_id: String,
    pub commands: Vec<Command>,
    pub observe: Vec<Assignment>,
}
pub fn work(cluster: &Cluster, node: &str) -> Result<Work, String> {
    Ok(Work {
        controller_id: cluster
            .controller_id
            .clone()
            .ok_or("missing controller identity")?,
        commands: commands(cluster, node)?,
        observe: cluster
            .executions
            .iter()
            .filter(|(_, e)| {
                e.members
                    .get(node)
                    .is_some_and(|m| m.phase != Phase::Exited)
            })
            .map(|(id, e)| Assignment {
                job: id.clone(),
                attempt: e.attempt,
            })
            .collect(),
    })
}

fn reserve_rendezvous(
    cluster: &Cluster,
    job: &str,
    placement: &[Placement],
) -> Result<Option<Rendezvous>, String> {
    if placement.len() <= 1 {
        return Ok(None);
    }
    let address = cluster.nodes[&placement[0].node]
        .address
        .clone()
        .ok_or("distributed placement requires training addresses")?;
    let used: BTreeSet<u16> = cluster
        .executions
        .values()
        .filter(|e| e.members.values().any(|m| m.phase != Phase::Exited))
        .filter_map(|e| e.rendezvous.as_ref())
        .filter(|r| r.address == address)
        .map(|r| r.port)
        .collect();
    let port = (29400..29500)
        .find(|port| !used.contains(port))
        .ok_or("no free rendezvous port on master address")?;
    Ok(Some(Rendezvous {
        address,
        port,
        id: format!("{}-{}", job, cluster.jobs[job].attempts.len() + 1),
    }))
}
