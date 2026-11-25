use std::{
    io::Read,
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use soul::{
    api::{self, Service},
    model::*,
    scheduler,
    store::Store,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const HELP: &str = "\
Soul GPU workload scheduler

Usage:
  soul example <inventory|job|trace>                      Print a bundled JSON example
  soul validate INVENTORY.json                            Check an inventory offline
  soul simulate INVENTORY.json                            Run two example scheduling decisions
  soul replay TRACE.json                                  Compare policies on a workload trace
  soul serve-sim DATABASE INVENTORY.json [127.0.0.1:PORT]  Start the simulation API
  soul serve DATABASE INVENTORY.json [127.0.0.1:PORT]      Start the managed controller
  soul --version                                          Print the installed version

The default listener is 127.0.0.1:8080. Managed mode requires SOUL_TOKEN.";

fn inventory(path: &str) -> Result<Cluster> {
    let input = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let cluster: Cluster = serde_json::from_slice(&input).map_err(|e| format!("{path}: {e}"))?;
    if !cluster.jobs.is_empty()
        || !cluster.executions.is_empty()
        || !cluster.heartbeats.is_empty()
        || !cluster.checkpoint_nodes.is_empty()
        || cluster.controller_id.is_some()
    {
        return Err("initial inventory must be free of controller and workload state".into());
    }
    cluster.validate()?;
    Ok(cluster)
}

fn simulate(mut cluster: Cluster) -> Result<()> {
    cluster.mode = Mode::Simulation;
    let pool = cluster.pools.keys().next().ok_or("no pools")?.clone();
    let account = cluster
        .accounts
        .values()
        .find(|a| {
            a.pool == pool
                && !cluster
                    .accounts
                    .values()
                    .any(|child| child.parent.as_deref() == Some(&a.id))
        })
        .ok_or("no leaf account")?
        .id
        .clone();
    for (id, protected) in [
        ("opportunistic", 0),
        ("training", 60.min(cluster.max_protected_seconds)),
    ] {
        cluster.submit(JobSpec {
            id: id.into(),
            pool: pool.clone(),
            account: account.clone(),
            image: "example/training".into(),
            command: vec!["train".into()],
            nodes: 1,
            fallback_nodes: vec![],
            gpus_per_node: 1,
            protected_seconds: protected,
            resumable: true,
            resize_policy: None,
            same_gpu_domain: false,
        })?;
        println!(
            "{}",
            serde_json::to_string_pretty(&scheduler::simulate(&mut cluster, &pool))?
        );
        cluster.advance(cluster.now.saturating_add(10))?;
    }
    Ok(())
}

async fn serve(
    database: &str,
    mut initial: Cluster,
    managed: bool,
    bind: Option<&str>,
) -> Result<()> {
    let token = if managed {
        let token =
            std::env::var("SOUL_TOKEN").map_err(|_| "set SOUL_TOKEN for managed mode")?;
        if token.len() < 16 {
            return Err("SOUL_TOKEN must be at least 16 characters".into());
        }
        initial.mode = Mode::Managed;
        let mut random = [0u8; 16];
        std::fs::File::open("/dev/urandom")?.read_exact(&mut random)?;
        initial.controller_id = Some(random.iter().map(|byte| format!("{byte:02x}")).collect());
        Some(token)
    } else {
        initial.mode = Mode::Simulation;
        None
    };
    let address: SocketAddr = bind.unwrap_or("127.0.0.1:8080").parse()?;
    if !address.ip().is_loopback() {
        return Err(
            "this release requires loopback; use an authenticated tunnel for remote nodes".into(),
        );
    }
    let listener = tokio::net::TcpListener::bind(address).await?;
    let address = listener.local_addr()?;
    let store = Arc::new(Mutex::new(Store::open(database, initial)?));
    let service = Service { store, token };
    let app = api::router(service.clone());
    if managed {
        tokio::spawn(async move {
            loop {
                if let Err(error) = api::tick(&service) {
                    eprintln!("controller reconciliation: {error}");
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        });
    }
    eprintln!(
        "Soul {} API on {address}",
        if managed { "managed" } else { "simulation" }
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let args: Vec<_> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["--help" | "-h"] => println!("{HELP}"),
        ["--version" | "-V"] => println!("soul {}", env!("CARGO_PKG_VERSION")),
        ["example", name] => {
            let json = match *name {
                "inventory" => include_str!("../examples/small.json"),
                "job" => include_str!("../examples/job.json"),
                "trace" => include_str!("../examples/resize-trace.json"),
                _ => return Err("expected example inventory, job, or trace".into()),
            };
            print!("{json}");
        }
        ["validate", path] => {
            inventory(path)?;
            println!("Valid inventory: {path}");
        }
        ["simulate", path] => simulate(inventory(path)?)?,
        ["replay", path] => {
            let trace: soul::replay::Trace = serde_json::from_reader(std::fs::File::open(path)?)?;
            println!("{}", serde_json::to_string_pretty(&trace.compare()?)?);
        }
        [command @ ("serve" | "serve-sim"), database, path, bind @ ..] if bind.len() <= 1 => {
            serve(
                database,
                inventory(path)?,
                *command == "serve",
                bind.first().copied(),
            )
            .await?;
        }
        _ => {
            eprintln!("{HELP}");
            std::process::exit(2);
        }
    }
    Ok(())
}
