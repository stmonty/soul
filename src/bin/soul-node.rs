//! A blocking, single-worker node daemon. Runtime state is recovered from Docker.
use std::time::Duration;
use soul::{
    execution::{Report, Work},
    node::{Docker, Journal},
};

const HELP: &str = "\
Soul Docker node daemon

Usage:
  soul-node http://127.0.0.1:PORT NODE JOURNAL.sqlite [--cpu-test] [--host-network]
  soul-node --version

Set SOUL_TOKEN to the managed controller credential.
--cpu-test      Execute against synthetic inventory with GPU attachment disabled.
--host-network  Use the host network for trusted multi-node or checkpoint-capable jobs.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if let [_, flag] = args.as_slice() {
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--version" | "-V" => {
                println!("soul-node {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            _ => {}
        }
    }
    if !(4..=6).contains(&args.len())
        || args[4..]
            .iter()
            .any(|arg| arg != "--cpu-test" && arg != "--host-network")
    {
        eprintln!("{HELP}");
        std::process::exit(2);
    }
    let endpoint = args[1].trim_end_matches('/');
    let address: std::net::SocketAddr = endpoint
        .strip_prefix("http://")
        .ok_or("use loopback HTTP through an authenticated tunnel")?
        .parse()?;
    if !address.ip().is_loopback() {
        return Err("node HTTP transport requires loopback; tunnel remote connections".into());
    }
    let node = &args[2];
    if !soul::model::valid_id(node) {
        return Err("invalid node identity".into());
    }
    let token = std::env::var("SOUL_TOKEN").map_err(|_| "set SOUL_TOKEN")?;
    let authorization = format!("Bearer {token}");
    let client: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(15)))
        .build()
        .into();
    let mut journal: Option<Journal> = None;
    let mut namespace = None;
    let mut docker = Docker {
        controller_url: endpoint.to_string(),
        cpu_test: args[4..].iter().any(|arg| arg == "--cpu-test"),
        host_network: args[4..].iter().any(|arg| arg == "--host-network"),
    };
    loop {
        let cycle = || -> Result<Work, Box<dyn std::error::Error>> {
            Ok(client
                .post(&format!("{endpoint}/v1/nodes/{node}/work"))
                .header("authorization", &authorization)
                .header(
                    "x-soul-checkpoint-version",
                    if docker.host_network { "1" } else { "0" },
                )
                .send_empty()?
                .body_mut()
                .read_json()?)
        };
        match cycle() {
            Ok(work) => {
                let identity = format!("{}-{node}", work.controller_id);
                if namespace
                    .as_ref()
                    .is_some_and(|previous| previous != &identity)
                {
                    return Err(
                        "controller identity changed; refusing to reuse node journal".into(),
                    );
                }
                if journal.is_none() {
                    journal = Some(Journal::open(&args[3], identity.clone())?);
                    namespace = Some(identity);
                }
                let journal = journal.as_ref().unwrap();
                let send = |report: Report| {
                    if let Err(error) = client
                        .post(&format!("{endpoint}/v1/nodes/{node}/report"))
                        .header("authorization", &authorization)
                        .send_json(&report)
                    {
                        eprintln!("report {} attempt {}: {error}", report.job, report.attempt);
                    }
                };
                for command in work.commands {
                    match journal.execute(&command, &mut docker) {
                        Ok(report) => send(report),
                        Err(error) => eprintln!(
                            "execute {} attempt {}: {error}",
                            command.job, command.attempt
                        ),
                    }
                }
                for assignment in work.observe {
                    match journal.observe(&assignment.job, assignment.attempt, &mut docker) {
                        Ok(Some(report)) => send(report),
                        Ok(None) => {}
                        Err(error) => eprintln!("observe {}: {error}", assignment.job),
                    }
                }
            }
            Err(error) => {
                eprintln!("controller unavailable: {error}; keeping local execution intact")
            }
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}
