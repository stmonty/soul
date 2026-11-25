use soul::replay::Trace;

#[test]
fn replay_accounts_for_checkpoint_cost_and_effect_on_competing_gang() {
    let fixture = include_str!("../examples/resize-trace.json");
    let trace: Trace = serde_json::from_str(fixture).unwrap();
    let runs = trace.compare().unwrap();
    let times: Vec<_> = runs
        .iter()
        .map(|r| (r.jobs["trainer"].finished_at, r.jobs["gang"].queued_seconds))
        .collect();
    assert_eq!(
        times,
        [
            (Some(34), 18),
            (Some(49), 33),
            (Some(33), 17),
            (Some(33), 17)
        ]
    );
    let trainer = &runs[2].jobs["trainer"];
    assert_eq!(
        (
            trainer.checkpoint_seconds,
            trainer.restore_seconds,
            trainer.gpu_seconds
        ),
        (2, 1, 53)
    );
    assert_eq!(runs[2].fair_share.accounts["train"], 53);

    let mut input: serde_json::Value = serde_json::from_str(fixture).unwrap();
    input["workloads"][1]["restore_seconds"] = 60.into();
    let costly: Trace = serde_json::from_value(input.clone()).unwrap();
    let costly = costly.compare().unwrap();
    assert!(costly[2].jobs["trainer"].finished_at > costly[1].jobs["trainer"].finished_at);
    assert_eq!(costly[3].jobs["trainer"].attempts, 1);
    assert_eq!(costly[3].jobs["trainer"].finished_at, Some(49));
    assert_eq!(costly[3].jobs["gang"].queued_seconds, 33);
    input["horizon"] = 17.into();
    let short: Trace = serde_json::from_value(input).unwrap();
    assert!(
        short
            .compare()
            .unwrap()
            .iter()
            .all(|r| r.jobs["trainer"].finished_at.is_none())
    );
}

#[test]
fn replay_counts_preemptions_and_requeued_time() {
    let mut input: serde_json::Value =
        serde_json::from_str(include_str!("../examples/resize-trace.json")).unwrap();
    input["workloads"][0]["spec"]["protected_seconds"] = 0.into();
    input["workloads"][0]["spec"]["resumable"] = true.into();
    let trace: Trace = serde_json::from_value(input).unwrap();
    let runs = trace.compare().unwrap();
    let blocker = &runs[0].jobs["blocker"];
    assert_eq!(
        (blocker.progress, blocker.preemptions, blocker.attempts),
        (10, 1, 2)
    );
    assert!(blocker.queued_seconds > 0);
}

#[test]
fn replay_charges_actual_restore_cost_even_when_the_forecast_is_optimistic() {
    let mut input: serde_json::Value =
        serde_json::from_str(include_str!("../examples/resize-trace.json")).unwrap();
    input["workloads"][1]["restore_seconds"] = 60.into();
    input["workloads"][1]["forecast"] =
        serde_json::json!({"rates":{"1":1,"2":2},"checkpoint_seconds":2,"restore_seconds":1});
    let trace: Trace = serde_json::from_value(input).unwrap();
    let runs = trace.compare().unwrap();
    let trainer = &runs[3].jobs["trainer"];
    assert_eq!(runs[3].policy, "cost_aware");
    assert_eq!((trainer.attempts, trainer.restore_seconds), (2, 60));
    assert_eq!(trainer.finished_at, runs[2].jobs["trainer"].finished_at);
    assert_eq!(trainer.finished_at, Some(92));
    let growth = &trainer.growth_estimates[0];
    assert_eq!(
        (
            growth.requested_at,
            growth.prediction.resize_seconds,
            growth.observed_seconds_to_finish
        ),
        (10, 23, Some(82))
    );
}
