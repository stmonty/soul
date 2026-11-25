use soul::{
    execution::{self, Command, Operation, Phase, Report, StopReason},
    model::*,
    node::{ContainerState, Journal, Runtime},
    store::Store,
};
use std::collections::BTreeMap;

fn cluster(nodes: u32) -> Cluster {
    let mut cluster: Cluster =
        serde_json::from_str(include_str!("../examples/small.json")).unwrap();
    cluster.mode = Mode::Managed;
    cluster.controller_id = Some("test-controller".into());
    for n in 1..=nodes {
        let id = format!("node-{n}");
        cluster.nodes.insert(
            id.clone(),
            Node {
                address: Some("127.0.0.1".into()),
                id: id.clone(),
                pool: "b300".into(),
                gpus: 1,
                schedulable: true,
                gpu_domains: Default::default(),
            },
        );
        execution::heartbeat(&mut cluster, &id).unwrap();
    }
    cluster
}
fn submit(cluster: &mut Cluster, id: &str, nodes: u32, account: &str) {
    cluster
        .submit(JobSpec {
            id: id.into(),
            pool: "b300".into(),
            account: account.into(),
            image: "ubuntu:24.04".into(),
            command: vec!["true".into()],
            nodes,
            fallback_nodes: vec![],
            gpus_per_node: 1,
            protected_seconds: 10,
            resumable: true,
            resize_policy: None,
            same_gpu_domain: false,
        })
        .unwrap();
}
fn report(
    cluster: &mut Cluster,
    node: &str,
    job: &str,
    attempt: u32,
    phase: Phase,
    exit_code: Option<i32>,
) -> Result<(), String> {
    execution::report(
        cluster,
        node,
        Report {
            job: job.into(),
            attempt,
            phase,
            exit_code,
        },
    )?;
    cluster.validate()
}
fn running(cluster: &mut Cluster, id: &str) {
    execution::reconcile(cluster);
    report(cluster, "node-1", id, 1, Phase::Prepared, None).unwrap();
    report(cluster, "node-1", id, 1, Phase::Running, None).unwrap();
}

#[test]
fn preparation_barrier_and_reservations_prevent_early_start() {
    let mut c = cluster(2);
    submit(&mut c, "gang", 2, "training");
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["gang"].state, JobState::Starting);
    report(&mut c, "node-1", "gang", 1, Phase::Prepared, None).unwrap();
    assert!(execution::commands(&c, "node-1").unwrap().is_empty());
    assert!(report(&mut c, "node-1", "gang", 1, Phase::Running, None).is_err());
    submit(&mut c, "waiting", 1, "rollouts");
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["waiting"].state, JobState::Queued);
    report(&mut c, "node-2", "gang", 1, Phase::Prepared, None).unwrap();
    assert_eq!(
        execution::commands(&c, "node-1").unwrap()[0].operation,
        Operation::Start
    );
    report(&mut c, "node-1", "gang", 1, Phase::Running, None).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Starting);
    report(&mut c, "node-2", "gang", 1, Phase::Running, None).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Running);
}

#[test]
fn cancellation_waits_for_every_node_and_is_idempotent() {
    let mut c = cluster(2);
    submit(&mut c, "gang", 2, "training");
    execution::reconcile(&mut c);
    execution::request_stop(&mut c, "gang", StopReason::Cancel).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Stopping);
    report(&mut c, "node-1", "gang", 1, Phase::Exited, Some(0)).unwrap();
    report(&mut c, "node-1", "gang", 1, Phase::Exited, Some(0)).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Stopping);
    assert_eq!(c.jobs["gang"].attempts[0].ended_at, None);
    report(&mut c, "node-2", "gang", 1, Phase::Exited, Some(143)).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Cancelled);
    assert_eq!(c.jobs["gang"].attempts[0].ended_at, Some(0));
}

#[test]
fn preemption_keeps_resources_until_confirmed_and_fences_old_reports() {
    let mut c = cluster(1);
    submit(&mut c, "train", 1, "training");
    running(&mut c, "train");
    c.advance(11).unwrap();
    submit(&mut c, "rollout", 1, "rollouts");
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["train"].state, JobState::Stopping);
    assert_eq!(c.jobs["rollout"].state, JobState::Queued);
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["rollout"].state, JobState::Queued);
    report(&mut c, "node-1", "train", 1, Phase::Exited, Some(143)).unwrap();
    running(&mut c, "rollout");
    report(&mut c, "node-1", "rollout", 1, Phase::Exited, Some(0)).unwrap();
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["train"].attempts.len(), 2);
    assert!(report(&mut c, "node-1", "train", 1, Phase::Exited, Some(143)).is_err());
    assert_eq!(c.jobs["train"].state, JobState::Starting);
}

#[test]
fn lost_heartbeat_never_releases_running_allocation() {
    let mut c = cluster(1);
    submit(&mut c, "train", 1, "training");
    running(&mut c, "train");
    c.advance(1000).unwrap();
    submit(&mut c, "new", 1, "rollouts");
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["train"].state, JobState::Running);
    assert_eq!(c.jobs["new"].state, JobState::Queued);
    assert_eq!(c.jobs["train"].attempts[0].ended_at, None);
}

#[test]
fn drained_node_keeps_its_allocation_until_the_whole_gang_exits() {
    let mut c = cluster(2);
    submit(&mut c, "gang", 2, "training");
    execution::reconcile(&mut c);
    c.nodes.get_mut("node-1").unwrap().schedulable = false;
    execution::request_stop(&mut c, "gang", StopReason::Cancel).unwrap();
    report(&mut c, "node-1", "gang", 1, Phase::Exited, Some(0)).unwrap();
    c.advance(1000).unwrap();
    let drain = c.node_scheduling("node-1").unwrap();
    assert!(!drain.schedulable);
    assert_eq!(drain.allocations.len(), 1);
    assert_eq!(drain.allocations[0].job, "gang");
    report(&mut c, "node-2", "gang", 1, Phase::Exited, Some(143)).unwrap();
    assert!(c.node_scheduling("node-1").unwrap().allocations.is_empty());
    submit(&mut c, "next", 2, "training");
    for node in ["node-1", "node-2"] {
        execution::heartbeat(&mut c, node).unwrap();
    }
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["next"].state, JobState::Queued);
    c.nodes.get_mut("node-1").unwrap().schedulable = true;
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["next"].state, JobState::Starting);
}

#[test]
fn startup_timeout_stops_partial_launch_instead_of_releasing_it() {
    let mut c = cluster(2);
    submit(&mut c, "gang", 2, "training");
    execution::reconcile(&mut c);
    report(&mut c, "node-1", "gang", 1, Phase::Prepared, None).unwrap();
    c.advance(execution::STARTUP_TIMEOUT).unwrap();
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["gang"].state, JobState::Stopping);
    assert_eq!(
        execution::commands(&c, "node-2").unwrap()[0].operation,
        Operation::Stop
    );
    report(&mut c, "node-1", "gang", 1, Phase::Exited, Some(0)).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Stopping);
    report(&mut c, "node-2", "gang", 1, Phase::Exited, Some(255)).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Failed);
}

#[test]
fn partial_failure_stops_peers_and_exit_code_is_required() {
    let mut c = cluster(2);
    submit(&mut c, "gang", 2, "training");
    execution::reconcile(&mut c);
    assert!(report(&mut c, "node-1", "gang", 1, Phase::Exited, None).is_err());
    report(&mut c, "node-1", "gang", 1, Phase::Exited, Some(1)).unwrap();
    assert_eq!(
        execution::commands(&c, "node-2").unwrap()[0].operation,
        Operation::Stop
    );
    report(&mut c, "node-2", "gang", 1, Phase::Exited, Some(0)).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Failed);
}

#[test]
fn protection_begins_after_readiness_not_during_container_setup() {
    let mut c = cluster(1);
    submit(&mut c, "train", 1, "training");
    execution::reconcile(&mut c);
    c.advance(20).unwrap();
    report(&mut c, "node-1", "train", 1, Phase::Prepared, None).unwrap();
    report(&mut c, "node-1", "train", 1, Phase::Running, None).unwrap();
    submit(&mut c, "rollout", 1, "rollouts");
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["train"].state, JobState::Running);
    c.advance(30).unwrap();
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["train"].state, JobState::Stopping);
}

#[test]
fn controller_restart_reissues_same_command_and_rejects_simulation_finish() {
    let path = std::env::temp_dir().join(format!("soul-controller-{}.sqlite", std::process::id()));
    let command;
    {
        let mut store = Store::open(&path, cluster(2)).unwrap();
        store
            .update(|c| {
                submit(c, "train", 2, "training");
                execution::reconcile(c);
                Ok(())
            })
            .unwrap();
        command = execution::commands(store.state(), "node-1").unwrap();
        assert!(store.update(|c| c.finish("train", true)).is_err());
    }
    {
        let store = Store::open(&path, cluster(2)).unwrap();
        assert_eq!(
            command,
            execution::commands(store.state(), "node-1").unwrap()
        );
    }
    std::fs::remove_file(path).unwrap();
}

#[derive(Default)]
struct FakeRuntime {
    states: BTreeMap<String, ContainerState>,
    starts: usize,
    fail_after_start: bool,
    unavailable: bool,
}
impl Runtime for FakeRuntime {
    fn inspect(&mut self, name: &str) -> Result<Option<ContainerState>, String> {
        if self.unavailable {
            return Err("runtime unavailable".into());
        }
        Ok(self.states.get(name).cloned())
    }
    fn prepare(&mut self, name: &str, _: &Operation) -> Result<(), String> {
        self.states.insert(name.into(), ContainerState::Created);
        Ok(())
    }
    fn start(&mut self, name: &str) -> Result<(), String> {
        self.starts += 1;
        self.states.insert(name.into(), ContainerState::Running);
        if self.fail_after_start {
            self.fail_after_start = false;
            return Err("lost start acknowledgement".into());
        }
        Ok(())
    }
    fn stop(&mut self, name: &str) -> Result<(), String> {
        self.states.insert(name.into(), ContainerState::Exited(143));
        Ok(())
    }
}
fn prepare() -> Command {
    Command {
        job: "train".into(),
        attempt: 1,
        operation: Operation::Prepare {
            image: "example".into(),
            command: vec!["train".into()],
            gpus: vec![0],
            rank: 0,
            nodes: vec!["node-1".into()],
            rendezvous: None,
            runtime: None,
            restore: None,
        },
    }
}

#[test]
fn pre_rename_attempt_keeps_its_original_container_after_restart() {
    let path = std::env::temp_dir().join(format!("soul-legacy-{}.sqlite", std::process::id()));
    drop(Journal::open(&path, "node".into()).unwrap());
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute(
        "INSERT INTO attempts VALUES ('train', 1, ?1)",
        [r#"{"phase":"running","exit_code":null}"#],
    )
    .unwrap();
    drop(db);

    let mut runtime = FakeRuntime::default();
    runtime
        .states
        .insert("wizard-node-train-1".into(), ContainerState::Running);
    let journal = Journal::open(&path, "node".into()).unwrap();
    assert_eq!(
        journal
            .observe("train", 1, &mut runtime)
            .unwrap()
            .unwrap()
            .phase,
        Phase::Running
    );
    let stop = Command {
        operation: Operation::Stop,
        ..prepare()
    };
    assert_eq!(
        journal.execute(&stop, &mut runtime).unwrap().exit_code,
        Some(143)
    );
    journal
        .execute(
            &Command {
                attempt: 2,
                ..prepare()
            },
            &mut runtime,
        )
        .unwrap();
    assert_eq!(runtime.states["soul-node-train-2"], ContainerState::Created);
    drop(journal);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn node_restart_after_lost_start_ack_does_not_restart_container() {
    let path = std::env::temp_dir().join(format!("soul-node-{}.sqlite", std::process::id()));
    let mut runtime = FakeRuntime::default();
    let command = Command {
        operation: Operation::Start,
        ..prepare()
    };
    {
        let journal = Journal::open(&path, "test".into()).unwrap();
        journal.execute(&prepare(), &mut runtime).unwrap();
        runtime.fail_after_start = true;
        assert!(journal.execute(&command, &mut runtime).is_err());
    }
    {
        let journal = Journal::open(&path, "test".into()).unwrap();
        assert_eq!(
            journal.execute(&command, &mut runtime).unwrap().phase,
            Phase::Running
        );
        assert_eq!(runtime.starts, 1);
        let stop = Command {
            operation: Operation::Stop,
            ..prepare()
        };
        journal.execute(&stop, &mut runtime).unwrap();
        assert_eq!(
            journal.execute(&command, &mut runtime).unwrap().phase,
            Phase::Exited
        );
        assert_eq!(
            journal.execute(&prepare(), &mut runtime).unwrap().phase,
            Phase::Exited
        );
        assert_eq!(runtime.starts, 1);
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn unavailable_runtime_cannot_be_mistaken_for_confirmed_termination() {
    let journal = Journal::open(":memory:", "test".into()).unwrap();
    let mut runtime = FakeRuntime::default();
    journal.execute(&prepare(), &mut runtime).unwrap();
    journal
        .execute(
            &Command {
                operation: Operation::Start,
                ..prepare()
            },
            &mut runtime,
        )
        .unwrap();
    runtime.unavailable = true;
    assert!(
        journal
            .execute(
                &Command {
                    operation: Operation::Stop,
                    ..prepare()
                },
                &mut runtime
            )
            .is_err()
    );
    runtime.unavailable = false;
    assert_eq!(
        journal
            .observe("train", 1, &mut runtime)
            .unwrap()
            .unwrap()
            .phase,
        Phase::Running
    );
}

#[test]
fn a_successful_early_rank_does_not_leave_gang_in_startup_forever() {
    let mut c = cluster(2);
    submit(&mut c, "gang", 2, "training");
    execution::reconcile(&mut c);
    report(&mut c, "node-1", "gang", 1, Phase::Prepared, None).unwrap();
    report(&mut c, "node-2", "gang", 1, Phase::Prepared, None).unwrap();
    report(&mut c, "node-1", "gang", 1, Phase::Exited, Some(0)).unwrap();
    report(&mut c, "node-2", "gang", 1, Phase::Running, None).unwrap();
    assert_eq!(c.jobs["gang"].state, JobState::Running);
}

#[test]
fn distributed_attempts_reserve_ports_until_all_members_exit() {
    let mut c = cluster(4);
    submit(&mut c, "gang-a", 2, "training");
    submit(&mut c, "gang-b", 2, "training");
    execution::reconcile(&mut c);
    execution::reconcile(&mut c);
    let a = c.executions["gang-a"].rendezvous.clone().unwrap();
    let b = c.executions["gang-b"].rendezvous.clone().unwrap();
    assert_eq!(a.address, b.address);
    assert_ne!(a.port, b.port);
    assert_eq!(a.id, "gang-a-1");
    for node in ["node-1", "node-2"] {
        let command = execution::commands(&c, node).unwrap().remove(0);
        match command.operation {
            Operation::Prepare { rendezvous, .. } => assert_eq!(rendezvous, Some(a.clone())),
            _ => panic!("expected preparation"),
        }
    }
    execution::request_stop(&mut c, "gang-a", StopReason::Cancel).unwrap();
    report(&mut c, "node-1", "gang-a", 1, Phase::Exited, Some(0)).unwrap();
    assert!(
        c.executions["gang-a"]
            .members
            .values()
            .any(|m| m.phase != Phase::Exited)
    );
    report(&mut c, "node-2", "gang-a", 1, Phase::Exited, Some(0)).unwrap();
    submit(&mut c, "gang-c", 2, "training");
    execution::reconcile(&mut c);
    assert_eq!(
        c.executions["gang-c"].rendezvous.as_ref().unwrap().port,
        a.port
    );
}

#[test]
fn distributed_submission_requires_configured_training_addresses() {
    let mut c = cluster(2);
    c.nodes.get_mut("node-2").unwrap().address = None;
    let spec = JobSpec {
        id: "gang".into(),
        pool: "b300".into(),
        account: "training".into(),
        image: "train".into(),
        command: vec!["train".into()],
        nodes: 2,
        fallback_nodes: vec![],
        gpus_per_node: 1,
        protected_seconds: 60,
        resumable: true,
        resize_policy: None,
        same_gpu_domain: false,
    };
    assert!(c.submit(spec).is_err());
}

#[test]
fn managed_fallback_uses_single_node_launch_and_holds_charge_until_exit() {
    let mut c = cluster(2);
    for node in c.nodes.values_mut() {
        node.address = None;
        node.gpus = 4;
    }
    let mut spec: JobSpec = serde_json::from_str(include_str!("../examples/job.json")).unwrap();
    spec.nodes = 2;
    spec.fallback_nodes = vec![1];
    spec.gpus_per_node = 3;
    let id = spec.id.clone();
    c.submit(spec).unwrap();
    execution::reconcile(&mut c);
    let commands = execution::commands(&c, "node-1").unwrap();
    assert!(
        matches!(&commands[0].operation, Operation::Prepare { nodes, gpus, rendezvous: None, .. }
        if nodes == &["node-1"] && gpus == &[0, 1, 2])
    );
    assert!(execution::commands(&c, "node-2").unwrap().is_empty());
    c.advance(4).unwrap();
    execution::request_stop(&mut c, &id, StopReason::Cancel).unwrap();
    c.advance(40).unwrap();
    let charge = |c: &Cluster| soul::scheduler::pool_usage(c, "b300").accounts["training"];
    assert_eq!(charge(&c), 120);
    report(&mut c, "node-1", &id, 1, Phase::Exited, Some(143)).unwrap();
    c.advance(50).unwrap();
    assert_eq!(charge(&c), 120);
}

fn trainer(policy: soul::resize::Policy) -> Cluster {
    let mut c = cluster(2);
    c.checkpoint_nodes.insert("node-1".into());
    let mut spec: JobSpec = serde_json::from_str(include_str!("../examples/job.json")).unwrap();
    spec.id = "trainer".into();
    spec.nodes = 2;
    spec.fallback_nodes = vec![1];
    spec.protected_seconds = 2;
    spec.resize_policy = Some(policy);
    c.submit(spec).unwrap();
    running(&mut c, "trainer");
    c.checkpoint_nodes.insert("node-2".into());
    c
}

fn growth_request(id: &str) -> soul::resize::ResizeRequest {
    soul::resize::ResizeRequest {
        id: id.into(),
        nodes: 2,
        attempt: 1,
    }
}

#[test]
fn resize_decline_expiry_and_cancel_keep_old_attempt_fenced() {
    use soul::resize::{self, Outcome, Policy, Reply};
    let mut c = trainer(Policy::Manual);
    assert!(resize::request(&mut c, "trainer", growth_request("grow")).is_err());
    c.advance(3).unwrap();
    resize::request(&mut c, "trainer", growth_request("grow")).unwrap();
    resize::request(&mut c, "trainer", growth_request("grow")).unwrap();
    assert!(resize::request(&mut c, "trainer", growth_request("other")).is_err());
    assert!(resize::reply(&mut c, "trainer", 2, "grow", Reply::Declined).is_err());
    for _ in 0..2 {
        resize::reply(&mut c, "trainer", 1, "grow", Reply::Declined).unwrap();
    }
    resize::request(&mut c, "trainer", growth_request("again")).unwrap();
    c.advance(3 + resize::CHECKPOINT_TIMEOUT).unwrap();
    assert_eq!(
        c.jobs["trainer"].attempts[0].resizes[1].outcome,
        Outcome::Expired
    );
    assert!(resize::reply(&mut c, "trainer", 1, "again", Reply::Declined).is_err());
    assert_eq!(c.jobs["trainer"].state, JobState::Running);
    for node in ["node-1", "node-2"] {
        execution::heartbeat(&mut c, node).unwrap();
    }
    resize::request(&mut c, "trainer", growth_request("third")).unwrap();
    execution::request_stop(&mut c, "trainer", StopReason::Cancel).unwrap();
    assert!(resize::reply(&mut c, "trainer", 1, "third", Reply::Declined).is_err());
    assert_eq!(c.jobs["trainer"].attempts.len(), 1);
    c.validate().unwrap();
}

#[test]
fn checkpoint_waits_for_confirmed_exit_and_exact_target_across_restart() {
    use soul::resize::{self, Checkpoint, Policy, Reply};
    let mut c = trainer(Policy::GrowWhenIdle);
    c.advance(3).unwrap();
    execution::reconcile(&mut c);
    let checkpoint = Checkpoint {
        reference: "artifact-10".into(),
        progress: 10,
    };
    let ready = Reply::Ready {
        checkpoint: checkpoint.clone(),
    };
    resize::reply(&mut c, "trainer", 1, "auto-grow", ready.clone()).unwrap();
    resize::reply(&mut c, "trainer", 1, "auto-grow", ready).unwrap();
    c.advance(100).unwrap(); // disconnected old rank and target
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["trainer"].state, JobState::Stopping);
    assert_eq!(c.jobs["trainer"].attempts[0].ended_at, None);
    let path = std::env::temp_dir().join(format!("soul-resize-{}.sqlite", std::process::id()));
    drop(Store::open(&path, c.clone()).unwrap());
    let mut store = Store::open(&path, cluster(2)).unwrap();
    store
        .update(|c| {
            report(c, "node-1", "trainer", 1, Phase::Exited, Some(143))?;
            execution::heartbeat(c, "node-1")?;
            execution::reconcile(c);
            assert_eq!(c.jobs["trainer"].state, JobState::Queued);
            assert!(c.jobs["trainer"].attempts.len() == 1);
            execution::heartbeat(c, "node-2")?;
            execution::reconcile(c);
            let next = &c.jobs["trainer"].attempts[1];
            assert_eq!(next.placement.len(), 2);
            match &execution::commands(c, "node-2")?[0].operation {
                Operation::Prepare { restore, .. } => {
                    assert_eq!(restore.as_ref(), Some(&checkpoint))
                }
                _ => panic!("expected prepare"),
            }
            assert!(resize::reply(c, "trainer", 1, "auto-grow", Reply::Declined).is_err());
            Ok(())
        })
        .unwrap();
    drop(store);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn idle_growth_defers_to_queued_jobs_and_respects_runtime_decline() {
    use soul::resize::{self, Policy, Reply};
    let mut c = trainer(Policy::GrowWhenIdle);
    c.advance(3).unwrap();
    submit(&mut c, "other", 1, "rollouts");
    execution::reconcile(&mut c);
    assert_eq!(c.jobs["other"].state, JobState::Starting);
    assert!(c.jobs["trainer"].attempts[0].resizes.is_empty());
    execution::request_stop(&mut c, "other", StopReason::Cancel).unwrap();
    report(&mut c, "node-2", "other", 1, Phase::Exited, Some(0)).unwrap();
    execution::reconcile(&mut c);
    resize::reply(&mut c, "trainer", 1, "auto-grow", Reply::Declined).unwrap();
    for _ in 0..3 {
        assert!(execution::reconcile(&mut c).is_empty());
    }
    assert_eq!(c.jobs["trainer"].attempts[0].resizes.len(), 1);
    c.validate().unwrap();
}

#[test]
fn shrinking_requires_every_old_rank_to_exit_and_cancellation_wins() {
    use soul::resize::{self, Checkpoint, Policy, Reply, ResizeRequest};
    let mut c = cluster(2);
    c.checkpoint_nodes.extend(c.nodes.keys().cloned());
    let spec = JobSpec {
        nodes: 2,
        fallback_nodes: vec![1],
        protected_seconds: 0,
        resize_policy: Some(Policy::Manual),
        resumable: false,
        ..serde_json::from_str(include_str!("../examples/job.json")).unwrap()
    };
    let id = spec.id.clone();
    c.submit(spec).unwrap();
    execution::reconcile(&mut c);
    for phase in [Phase::Prepared, Phase::Running] {
        for node in ["node-1", "node-2"] {
            report(&mut c, node, &id, 1, phase.clone(), None).unwrap();
        }
    }
    resize::request(
        &mut c,
        &id,
        ResizeRequest {
            id: "shrink".into(),
            attempt: 1,
            nodes: 1,
        },
    )
    .unwrap();
    let ready = Reply::Ready {
        checkpoint: Checkpoint {
            reference: "durable/state".into(),
            progress: 7,
        },
    };
    resize::reply(&mut c, &id, 1, "shrink", ready.clone()).unwrap();
    report(&mut c, "node-1", &id, 1, Phase::Exited, Some(0)).unwrap();
    execution::reconcile(&mut c);
    assert_eq!(c.jobs[&id].state, JobState::Stopping);
    assert_eq!(c.jobs[&id].attempts[0].ended_at, None);

    let mut cancelled = c.clone();
    execution::request_stop(&mut cancelled, &id, StopReason::Cancel).unwrap();
    resize::reply(&mut cancelled, &id, 1, "shrink", ready).unwrap();
    report(&mut cancelled, "node-2", &id, 1, Phase::Exited, Some(0)).unwrap();
    execution::reconcile(&mut cancelled);
    assert_eq!(cancelled.jobs[&id].state, JobState::Cancelled);
    assert_eq!(cancelled.jobs[&id].attempts.len(), 1);

    report(&mut c, "node-2", &id, 1, Phase::Exited, Some(0)).unwrap();
    execution::reconcile(&mut c);
    assert_eq!(c.jobs[&id].attempts[1].placement.len(), 1);
    assert!(c.executions[&id].rendezvous.is_none());
    c.validate().unwrap();
}

#[test]
fn forecast_retries_do_not_refresh_expiry_or_replace_a_newer_report() {
    use soul::{
        forecast::{self, Assessment, Profile},
        resize::Policy,
    };
    let mut c = trainer(Policy::CostAware);
    c.advance(2).unwrap();
    let p = Profile {
        sequence: 1,
        valid_for_seconds: 3,
        remaining_work: 100,
        rates: [(1, 1), (2, 2)].into(),
        checkpoint_seconds: 2,
        restore_seconds: 1,
    };
    assert!(matches!(
        forecast::assess(&c, &c.jobs["trainer"], &c.jobs["trainer"].attempts[0]),
        Assessment::AwaitingProfile
    ));
    assert!(forecast::report(&mut c, "trainer", 2, p.clone()).is_err());
    forecast::report(&mut c, "trainer", 1, p.clone()).unwrap();
    let p2 = Profile {
        sequence: 2,
        ..p.clone()
    };
    forecast::report(&mut c, "trainer", 1, p2.clone()).unwrap();
    c.advance(5).unwrap();
    forecast::report(&mut c, "trainer", 1, p2.clone()).unwrap();
    assert!(forecast::report(&mut c, "trainer", 1, p).is_err());
    assert!(
        forecast::report(
            &mut c,
            "trainer",
            1,
            Profile {
                remaining_work: 10,
                ..p2.clone()
            }
        )
        .is_err()
    );
    assert!(matches!(
        forecast::assess(&c, &c.jobs["trainer"], &c.jobs["trainer"].attempts[0]),
        Assessment::Expired { sequence: 2 }
    ));
    assert!(execution::reconcile(&mut c).is_empty());
    let invalid = Profile {
        sequence: 3,
        rates: [(1, 1), (3, 1)].into(),
        ..p2.clone()
    };
    assert!(forecast::report(&mut c, "trainer", 1, invalid).is_err());
    forecast::report(&mut c, "trainer", 1, Profile { sequence: 3, ..p2 }).unwrap();
    c.advance(6).unwrap();
    execution::reconcile(&mut c);
    let e = c.jobs["trainer"].attempts[0].resizes[0]
        .estimate
        .as_ref()
        .unwrap();
    assert_eq!(
        (
            e.sequence,
            e.remaining_work,
            e.stay_seconds,
            e.resize_seconds
        ),
        (3, 99, 99, 53)
    );
    c.validate().unwrap();
}

#[test]
fn cost_aware_growth_can_prefer_an_intermediate_size_and_defers_to_queue() {
    use soul::{
        forecast::{self, Profile},
        resize::Policy,
    };
    let mut c = cluster(4);
    c.checkpoint_nodes.insert("node-1".into());
    let spec = JobSpec {
        nodes: 4,
        fallback_nodes: vec![2, 1],
        resize_policy: Some(Policy::CostAware),
        id: "trainer".into(),
        protected_seconds: 1,
        ..serde_json::from_str(include_str!("../examples/job.json")).unwrap()
    };
    c.submit(spec).unwrap();
    running(&mut c, "trainer");
    c.checkpoint_nodes.extend(c.nodes.keys().cloned());
    c.advance(2).unwrap();
    let p = Profile {
        sequence: 1,
        valid_for_seconds: 200,
        remaining_work: 100,
        rates: [(1, 1), (2, 8), (4, 8)].into(),
        checkpoint_seconds: 1,
        restore_seconds: 2,
    };
    forecast::report(&mut c, "trainer", 1, p).unwrap();
    let explain = soul::scheduler::explain(&c, &c.jobs["trainer"]);
    assert_eq!(
        explain.growth_forecast.unwrap().recommendation().unwrap().0,
        2
    );
    submit(&mut c, "other", 1, "rollouts");
    let decisions = execution::reconcile(&mut c);
    assert_eq!(decisions[0].job, "other");
    assert!(c.jobs["trainer"].attempts[0].resizes.is_empty());
    execution::reconcile(&mut c);
    let r = &c.jobs["trainer"].attempts[0].resizes[0];
    assert_eq!(r.request.nodes, 2);
    assert_eq!(r.estimate.as_ref().unwrap().resize_seconds, 16);
}

#[test]
fn cost_forecasts_require_a_feasible_strict_gain_within_the_checkpoint_deadline() {
    use soul::{
        forecast::{self, Assessment, Profile},
        resize::{CHECKPOINT_TIMEOUT, Policy},
    };
    let mut c = trainer(Policy::CostAware);
    c.advance(2).unwrap();
    let mut p = Profile {
        sequence: 0,
        valid_for_seconds: 10,
        remaining_work: 10,
        rates: [(1, 1)].into(),
        checkpoint_seconds: 0,
        restore_seconds: 0,
    };
    for (sequence, target_rate, checkpoint, restore) in [
        (1, None, 0, 0),
        (2, Some(2), 2, 3),
        (3, Some(2), 0, u64::MAX),
        (4, Some(2), CHECKPOINT_TIMEOUT, 0),
    ] {
        p.sequence = sequence;
        if let Some(rate) = target_rate {
            p.rates.insert(2, rate);
        }
        p.checkpoint_seconds = checkpoint;
        p.restore_seconds = restore;
        forecast::report(&mut c, "trainer", 1, p.clone()).unwrap();
        let job = &c.jobs["trainer"];
        let comparison = forecast::assess(&c, job, &job.attempts[0]);
        assert!(comparison.recommendation().is_none());
        if checkpoint == CHECKPOINT_TIMEOUT {
            assert!(matches!(
                comparison,
                Assessment::CheckpointExceedsDeadline { .. }
            ));
        }
        assert!(execution::reconcile(&mut c).is_empty());
    }
    c.checkpoint_nodes.remove("node-2");
    p.sequence += 1;
    p.checkpoint_seconds = 0;
    p.restore_seconds = 1;
    forecast::report(&mut c, "trainer", 1, p).unwrap();
    assert!(execution::reconcile(&mut c).is_empty());
    c.checkpoint_nodes.insert("node-2".into());
    assert_eq!(execution::reconcile(&mut c).len(), 1);
}
