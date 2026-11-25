use soul::{
    model::*,
    scheduler::{self, Action},
    store::Store,
};

fn cluster() -> Cluster {
    serde_json::from_str(include_str!("../examples/small.json")).unwrap()
}
fn job(id: &str, account: &str, protected: u64) -> JobSpec {
    JobSpec {
        id: id.into(),
        pool: "b300".into(),
        account: account.into(),
        image: "train".into(),
        command: vec!["train".into()],
        nodes: 1,
        fallback_nodes: vec![],
        gpus_per_node: 1,
        protected_seconds: protected,
        resumable: true,
        resize_policy: None,
        same_gpu_domain: false,
    }
}
fn started(c: &mut Cluster) -> String {
    let decision = scheduler::simulate(c, "b300").unwrap();
    assert!(matches!(decision.action, Action::Start { .. }));
    decision.job
}

#[test]
fn protected_job_yields_only_after_window() {
    let mut c = cluster();
    c.submit(job("train", "training", 60)).unwrap();
    assert_eq!(started(&mut c), "train");
    c.advance(10).unwrap();
    c.submit(job("rollout", "rollouts", 60)).unwrap();
    assert!(matches!(
        scheduler::next(&c, "b300").unwrap().action,
        Action::Wait
    ));
    let why = scheduler::explain(&c, &c.jobs["rollout"]);
    assert_eq!(
        why.allocation_owners["train"],
        scheduler::Reclaim::Protected { until: 60 }
    );
    c.advance(60).unwrap();
    assert_eq!(
        scheduler::explain(&c, &c.jobs["rollout"]).allocation_owners["train"],
        scheduler::Reclaim::Eligible
    );
    assert_eq!(started(&mut c), "rollout");
    assert_eq!(c.jobs["train"].state, JobState::Queued);
    assert_eq!(c.jobs["train"].attempts[0].ended_at, Some(60));
}

#[test]
fn funded_job_reclaims_opportunistic_capacity() {
    let mut c = cluster();
    c.submit(job("free", "training", 0)).unwrap();
    started(&mut c);
    c.submit(job("paid", "rollouts", 60)).unwrap();
    assert_eq!(started(&mut c), "paid");
}

#[test]
fn non_resumable_work_is_not_destroyed_to_admit_another_job() {
    let mut c = cluster();
    let mut spec = job("train", "training", 1);
    spec.resumable = false;
    c.submit(spec).unwrap();
    started(&mut c);
    c.advance(100).unwrap();
    c.submit(job("rollout", "rollouts", 1)).unwrap();
    assert!(matches!(
        scheduler::next(&c, "b300").unwrap().action,
        Action::Wait
    ));
}

#[test]
fn expired_usage_no_longer_penalizes_an_account() {
    let mut c = cluster();
    c.lookback_seconds = 100;
    let mut account = c.accounts.remove("rollouts").unwrap();
    account.id = "z-rollouts".into();
    c.accounts.insert(account.id.clone(), account);
    c.submit(job("old", "training", 1)).unwrap();
    started(&mut c);
    c.advance(10).unwrap();
    c.finish("old", false).unwrap();
    c.submit(job("first", "training", 1)).unwrap();
    c.advance(11).unwrap();
    c.submit(job("second", "z-rollouts", 1)).unwrap();
    assert_eq!(scheduler::next(&c, "b300").unwrap().job, "second");
    c.advance(111).unwrap();
    assert_eq!(scheduler::next(&c, "b300").unwrap().job, "first");
}

#[test]
fn fallback_uses_free_capacity_and_retries_charge_each_actual_gang() {
    let mut c: Cluster = serde_json::from_str(include_str!("../examples/two-node.json")).unwrap();
    c.accounts = cluster().accounts;
    for account in c.accounts.values_mut() {
        account.pool = "test".into();
    }
    let spec = |id, account, protection| JobSpec {
        pool: "test".into(),
        ..job(id, account, protection)
    };
    c.submit(spec("filler", "rollouts", 0)).unwrap();
    scheduler::simulate(&mut c, "test").unwrap();
    let trainer = JobSpec {
        nodes: 2,
        fallback_nodes: vec![1],
        ..spec("trainer", "training", 5)
    };
    c.submit(trainer).unwrap();
    let first = scheduler::simulate(&mut c, "test").unwrap();
    assert!(
        matches!(first.action, Action::Start { placement, preempt } if placement.len() == 1 && preempt.is_empty())
    );
    c.advance(10).unwrap();
    c.finish("filler", false).unwrap();
    c.submit(JobSpec {
        nodes: 2,
        ..spec("gang", "rollouts", 1)
    })
    .unwrap();
    assert_eq!(scheduler::simulate(&mut c, "test").unwrap().job, "gang");
    c.advance(12).unwrap();
    c.finish("gang", false).unwrap();
    assert_eq!(scheduler::simulate(&mut c, "test").unwrap().job, "trainer");
    assert_eq!(c.jobs["trainer"].attempts[1].placement.len(), 2);
    c.advance(26).unwrap();
    c.validate().unwrap();
    assert_eq!(scheduler::pool_usage(&c, "test").accounts["training"], 38);
    c.lookback_seconds = 20;
    let usage = scheduler::pool_usage(&c, "test");
    assert_eq!((usage.since, usage.as_of), (6, 26));
    assert_eq!(usage.accounts["training"], 32);
    assert_eq!(usage.accounts["rollouts"], 4);
    assert_eq!(usage.accounts["research"], 36);
}

#[test]
fn fallback_sizes_are_explicit_and_strictly_decreasing() {
    let mut c = cluster();
    let mut spec = job("trainer", "training", 1);
    spec.nodes = 4;
    for invalid in [vec![0], vec![4], vec![2, 2], vec![1, 2], vec![5]] {
        spec.fallback_nodes = invalid;
        assert!(
            c.submit(spec.clone())
                .unwrap_err()
                .contains("fallback_nodes")
        );
    }
    spec.fallback_nodes = vec![2];
    assert!(c.submit(spec.clone()).unwrap_err().contains("topology"));
    spec.fallback_nodes.push(1);
    c.submit(spec.clone()).unwrap();
    c.submit(spec).unwrap();
    started(&mut c);
    c.validate().unwrap();
}

#[test]
fn fallback_preemption_reclaims_only_devices_in_the_chosen_gang() {
    let mut c = cluster();
    let mut large = c.nodes["node-1"].clone();
    large.id = "node-2".into();
    large.gpus = 2;
    c.nodes.insert(large.id.clone(), large);
    c.submit(job("z-small", "training", 0)).unwrap();
    started(&mut c);
    c.submit(JobSpec {
        gpus_per_node: 2,
        ..job("a-large", "training", 0)
    })
    .unwrap();
    started(&mut c);
    c.submit(JobSpec {
        nodes: 2,
        fallback_nodes: vec![1],
        gpus_per_node: 2,
        ..job("trainer", "rollouts", 1)
    })
    .unwrap();
    let decision = scheduler::simulate(&mut c, "b300").unwrap();
    assert!(matches!(decision.action, Action::Start { preempt, .. } if preempt == ["a-large"]));
    assert_eq!(c.jobs["z-small"].state, JobState::Running);
    c.validate().unwrap();
}

#[test]
fn three_free_nodes_yield_a_declared_two_node_gang() {
    let mut c = cluster();
    for id in ["node-2", "node-3"] {
        let node = Node {
            id: id.into(),
            ..c.nodes["node-1"].clone()
        };
        c.nodes.insert(id.into(), node);
    }
    c.submit(JobSpec {
        nodes: 4,
        fallback_nodes: vec![2, 1],
        ..job("trainer", "training", 1)
    })
    .unwrap();
    started(&mut c);
    let placement = &mut c.jobs.get_mut("trainer").unwrap().attempts[0].placement;
    assert_eq!(placement.len(), 2);
    placement.push(Placement {
        node: "node-3".into(),
        gpus: vec![0],
    });
    assert!(c.validate().unwrap_err().contains("gang size"));
}

#[test]
fn parent_usage_controls_children_and_weights_are_respected() {
    let mut c = cluster();
    c.accounts.get_mut("training").unwrap().weight = 1;
    c.accounts.insert(
        "child".into(),
        Account {
            id: "child".into(),
            pool: "b300".into(),
            parent: Some("training".into()),
            weight: 1,
        },
    );
    c.submit(job("old", "child", 1)).unwrap();
    started(&mut c);
    c.advance(10).unwrap();
    c.finish("old", false).unwrap();
    c.submit(job("a", "child", 1)).unwrap();
    c.submit(job("b", "rollouts", 1)).unwrap();
    assert_eq!(started(&mut c), "b");
    c.advance(15).unwrap();
    c.finish("b", false).unwrap();
    c.submit(job("c", "rollouts", 1)).unwrap();
    assert_eq!(scheduler::next(&c, "b300").unwrap().job, "c");
    c.accounts.get_mut("training").unwrap().weight = 3;
    assert_eq!(scheduler::next(&c, "b300").unwrap().job, "a");
}

#[test]
fn gang_placement_is_all_or_nothing_and_pool_local() {
    let mut c = cluster();
    c.nodes.insert(
        "node-2".into(),
        Node {
            address: Some("127.0.0.1".into()),
            id: "node-2".into(),
            pool: "b300".into(),
            gpus: 1,
            schedulable: false,
            gpu_domains: Default::default(),
        },
    );
    c.pools.insert(
        "h100".into(),
        Pool {
            id: "h100".into(),
            gpu_kind: "H100".into(),
        },
    );
    c.nodes.insert(
        "h-node".into(),
        Node {
            address: Some("127.0.0.1".into()),
            id: "h-node".into(),
            pool: "h100".into(),
            gpus: 8,
            schedulable: true,
            gpu_domains: Default::default(),
        },
    );
    let mut spec = job("gang", "training", 60);
    spec.nodes = 2;
    c.submit(spec).unwrap();
    assert!(matches!(
        scheduler::simulate(&mut c, "b300").unwrap().action,
        Action::Wait
    ));
    assert!(c.jobs["gang"].attempts.is_empty());
    c.nodes.get_mut("node-2").unwrap().schedulable = true;
    started(&mut c);
    let placements = &c.jobs["gang"].attempts[0].placement;
    assert_eq!(placements.len(), 2);
    assert!(placements.iter().all(|p| c.nodes[&p.node].pool == "b300"));
}

#[test]
fn submission_is_idempotent_and_conflicts_are_rejected() {
    let mut c = cluster();
    let spec = job("job", "training", 1);
    c.submit(spec.clone()).unwrap();
    c.submit(spec).unwrap();
    assert_eq!(c.jobs.len(), 1);
    assert!(c.submit(job("job", "training", 2)).is_err());
    assert!(c.advance(10).is_ok());
    assert!(c.advance(9).is_err());
}

#[test]
fn account_cycles_and_cross_pool_parents_are_rejected() {
    let mut c = cluster();
    c.accounts.get_mut("research").unwrap().parent = Some("training".into());
    assert!(c.validate().is_err());
    c.accounts.get_mut("research").unwrap().parent = None;
    c.pools.insert(
        "h100".into(),
        Pool {
            id: "h100".into(),
            gpu_kind: "H100".into(),
        },
    );
    c.accounts.get_mut("training").unwrap().pool = "h100".into();
    assert!(c.validate().is_err());
}

#[test]
fn repeated_scheduling_never_double_allocates_a_gpu() {
    let mut c = cluster();
    c.nodes.get_mut("node-1").unwrap().gpus = 8;
    for n in 0..24 {
        c.submit(job(
            &format!("job-{n:02}"),
            if n % 2 == 0 { "training" } else { "rollouts" },
            10,
        ))
        .unwrap();
    }
    for t in 0..100 {
        c.advance(t).unwrap();
        for _ in 0..24 {
            let Some(decision) = scheduler::simulate(&mut c, "b300") else {
                break;
            };
            if matches!(decision.action, Action::Wait) {
                break;
            }
        }
        let mut occupied = std::collections::BTreeSet::new();
        for j in c.jobs.values().filter(|j| j.state == JobState::Running) {
            for p in &j.attempts.last().unwrap().placement {
                for gpu in &p.gpus {
                    assert!(occupied.insert((&p.node, gpu)));
                }
            }
        }
        assert!(occupied.len() <= 8);
    }
}

#[test]
fn durable_state_survives_reopen_and_failed_updates_rollback() {
    let path = std::env::temp_dir().join(format!("soul-test-{}.sqlite", std::process::id()));
    {
        let mut store = Store::open(&path, cluster()).unwrap();
        store
            .update(|c| c.submit(job("durable", "training", 1)))
            .unwrap();
        let result: Result<(), String> = store.update(|c| {
            c.advance(50)?;
            Err("abort".into())
        });
        assert!(result.is_err());
        assert_eq!(store.state().now, 0);
    }
    {
        let store = Store::open(&path, cluster()).unwrap();
        assert!(store.state().jobs.contains_key("durable"));
        assert_eq!(store.state().now, 0);
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn infeasible_preemption_plan_leaves_running_jobs_untouched() {
    let mut c = cluster();
    c.nodes.insert(
        "node-2".into(),
        Node {
            address: Some("127.0.0.1".into()),
            id: "node-2".into(),
            pool: "b300".into(),
            gpus: 1,
            schedulable: false,
            gpu_domains: Default::default(),
        },
    );
    c.submit(job("free", "training", 0)).unwrap();
    started(&mut c);
    let mut spec = job("gang", "rollouts", 1);
    spec.nodes = 2;
    c.submit(spec).unwrap();
    assert!(matches!(
        scheduler::simulate(&mut c, "b300").unwrap().action,
        Action::Wait
    ));
    assert_eq!(c.jobs["free"].state, JobState::Running);
    assert_eq!(c.jobs["free"].attempts[0].ended_at, None);

    // Reclaiming the GPU hypothetically for the blocked gang must not make it
    // appear free when the planner examines the next candidate in the same pass.
    c.submit(job("small", "rollouts", 1)).unwrap();
    let next = scheduler::simulate(&mut c, "b300").unwrap();
    assert_eq!(next.job, "small");
    assert!(matches!(next.action, Action::Start { preempt, .. } if preempt == ["free"]));
    c.validate().unwrap();
}

#[test]
fn container_arguments_and_job_identity_are_validated_before_admission() {
    let mut c = cluster();
    let mut spec = job("valid", "training", 1);
    spec.image = "--privileged".into();
    assert!(c.submit(spec).is_err());
    let mut spec = job("valid", "training", 1);
    spec.command = vec!["".into()];
    assert!(c.submit(spec).is_err());
    assert!(c.submit(job("bad/id", "training", 1)).is_err());
    assert!(c.jobs.is_empty());
}

#[test]
fn interconnect_domains_control_admission_placement_and_pending_explanations() {
    let mut c = cluster();
    let node = c.nodes.get_mut("node-1").unwrap();
    node.gpus = 4;
    node.gpu_domains = [("a".into(), vec![0, 2]), ("b".into(), vec![1, 3])].into();
    c.validate().unwrap();
    let mut spec = job("gang", "training", 10);
    spec.same_gpu_domain = true;
    spec.gpus_per_node = 3;
    assert!(c.submit(spec.clone()).is_err());
    spec.gpus_per_node = 2;
    for id in ["f0", "f1", "f2", "f3"] {
        let mut filler = job(id, "training", 60);
        filler.resumable = false;
        c.submit(filler).unwrap();
        started(&mut c);
    }
    // Small jobs pack into a single domain before opening the next one.
    assert_eq!(c.jobs["f1"].attempts[0].placement[0].gpus, vec![2]);
    c.finish("f0", false).unwrap();
    c.finish("f2", false).unwrap();
    c.submit(spec).unwrap();
    let why = scheduler::explain(&c, &c.jobs["gang"]);
    assert_eq!(why.queue_rank, Some(1));
    assert_eq!(why.nodes["node-1"].free_gpus, Some(2));
    assert_eq!(why.nodes["node-1"].largest_free_domain, Some(1));
    assert!(why.placement_on_free_devices.is_none());
    assert!(matches!(why.pool_decision.unwrap().action, Action::Wait));
    c.finish("f1", false).unwrap();
    started(&mut c);
    assert_eq!(c.jobs["gang"].attempts[0].placement[0].gpus, vec![0, 2]);
    c.jobs.get_mut("gang").unwrap().attempts[0].placement[0].gpus = vec![0, 1];
    assert!(c.validate().is_err());
}

#[test]
fn inventory_rejects_overlapping_domains_and_missing_devices() {
    let mut c = cluster();
    let domains = &mut c.nodes.get_mut("node-1").unwrap().gpu_domains;
    domains.insert("a".into(), vec![0]);
    domains.insert("b".into(), vec![0]);
    assert!(c.validate().is_err());
    c.nodes
        .get_mut("node-1")
        .unwrap()
        .gpu_domains
        .insert("b".into(), vec![1]);
    assert!(c.validate().is_err());
}
