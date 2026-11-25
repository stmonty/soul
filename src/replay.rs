//! Deterministic event-driven trace replay with integer-second timestamps using the production placement policy.
use crate::{
    model::*,
    resize::{self, Checkpoint, Reply},
    scheduler::{self, Action},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
pub struct Trace {
    inventory: Cluster,
    horizon: u64,
    workloads: Vec<Workload>,
}

#[derive(Deserialize)]
struct Workload {
    at: Timestamp,
    spec: JobSpec,
    work: u64,
    /// Useful work units per second, keyed by node count.
    rates: BTreeMap<u32, u64>,
    #[serde(default)]
    checkpoint_seconds: u64,
    #[serde(default)]
    restore_seconds: u64,
    /// Optional application predictions. The rates and costs above still drive execution.
    #[serde(default)]
    forecast: Option<ReplayForecast>,
}

#[derive(Deserialize)]
struct ReplayForecast {
    rates: BTreeMap<u32, u64>,
    checkpoint_seconds: u64,
    restore_seconds: u64,
}

impl Workload {
    fn ready_at(&self, attempt: &Attempt) -> Timestamp {
        if attempt.number == 1 {
            attempt.started_at
        } else {
            attempt.started_at.saturating_add(self.restore_seconds)
        }
    }

    /// Account [previous, now) before applying arrivals, exits or resize replies at now.
    fn account_interval(
        &self,
        job: &Job,
        result: &mut JobResult,
        previous: Timestamp,
        now: Timestamp,
    ) {
        let elapsed = now - previous;
        match job.state {
            JobState::Queued => result.queued_seconds += elapsed,
            JobState::Running => {
                let attempt = job.attempts.last().unwrap();
                if attempt.pending_resize().is_some() {
                    result.checkpoint_seconds += elapsed;
                    return;
                }
                let restoring = now.min(self.ready_at(attempt)).saturating_sub(previous);
                result.restore_seconds += restoring;
                let rate = self.rates[&attempt.node_count()];
                result.progress = result
                    .progress
                    .saturating_add(rate.saturating_mul(elapsed - restoring))
                    .min(self.work);
            }
            _ => {}
        }
    }

    fn acknowledge_checkpoint(&self, cluster: &mut Cluster, progress: u64) -> Result<bool, String> {
        let id = &self.spec.id;
        let Some(pending) = cluster.jobs[id]
            .attempts
            .last()
            .and_then(Attempt::pending_resize)
        else {
            return Ok(false);
        };
        if cluster.now - pending.requested_at < self.checkpoint_seconds {
            return Ok(false);
        }
        let request = pending.request.clone();
        let checkpoint = Checkpoint {
            reference: format!("simulation/{id}/{}", request.attempt),
            progress,
        };
        resize::reply(
            cluster,
            id,
            request.attempt,
            &request.id,
            Reply::Ready { checkpoint },
        )?;
        Ok(true)
    }

    fn report_if_ready(&self, c: &mut Cluster, result: &JobResult) -> Result<(), String> {
        let Some(job) = c.jobs.get(&self.spec.id) else {
            return Ok(());
        };
        if job.spec.resize_policy != Some(resize::Policy::CostAware)
            || job.state != JobState::Running
        {
            return Ok(());
        }
        let a = job.attempts.last().unwrap();
        if a.profile.is_some() || !a.resizes.is_empty() || c.now < self.ready_at(a) {
            return Ok(());
        }
        let (rates, checkpoint_seconds, restore_seconds) = match &self.forecast {
            Some(f) => (&f.rates, f.checkpoint_seconds, f.restore_seconds),
            None => (&self.rates, self.checkpoint_seconds, self.restore_seconds),
        };
        let profile = crate::forecast::Profile {
            sequence: 1,
            remaining_work: self.work - result.progress,
            valid_for_seconds: u64::MAX,
            rates: rates
                .iter()
                .filter(|(n, _)| job.spec.supports_node_count(**n))
                .map(|(&n, &r)| (n, r))
                .collect(),
            checkpoint_seconds,
            restore_seconds,
        };
        let number = a.number;
        crate::forecast::report(c, &self.spec.id, number, profile)
    }
}

#[derive(Default, Serialize)]
pub struct JobResult {
    pub first_start: Option<Timestamp>,
    pub finished_at: Option<Timestamp>,
    pub queued_seconds: u64,
    pub checkpoint_seconds: u64,
    pub restore_seconds: u64,
    pub progress: u64,
    pub gpu_seconds: u128,
    pub attempts: usize,
    pub preemptions: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub growth_estimates: Vec<GrowthResult>,
}

#[derive(Serialize)]
pub struct GrowthResult {
    pub attempt: u32,
    pub requested_at: Timestamp,
    pub nodes: u32,
    pub prediction: crate::forecast::Estimate,
    /// Includes any subsequent waiting, preemption, or additional resizing.
    pub observed_seconds_to_finish: Option<u64>,
}

#[derive(Serialize)]
pub struct Run {
    pub policy: &'static str,
    pub as_of: Timestamp,
    pub jobs: BTreeMap<String, JobResult>,
    pub fair_share: scheduler::PoolUsage,
}

#[derive(Clone, Copy)]
enum Sizing {
    Fixed,
    Fallback,
    Adaptive,
    CostAware,
}

impl Sizing {
    fn job_spec(self, original: &JobSpec) -> JobSpec {
        let mut spec = original.clone();
        if matches!(self, Sizing::Fixed) {
            spec.fallback_nodes.clear();
        }
        spec.resize_policy = match (self, spec.resize_policy) {
            (Sizing::Fixed | Sizing::Fallback, _) => None,
            (_, None | Some(resize::Policy::Manual)) => spec.resize_policy,
            (Sizing::Adaptive, _) => Some(resize::Policy::GrowWhenIdle),
            (Sizing::CostAware, _) => Some(resize::Policy::CostAware),
        };
        spec
    }
}

impl Trace {
    pub fn compare(&self) -> Result<Vec<Run>, String> {
        self.inventory.validate()?;
        if self.inventory.mode != Mode::Simulation
            || self.inventory.now != 0
            || !self.inventory.jobs.is_empty()
            || self.inventory.pools.len() != 1
            || self.workloads.is_empty()
            || self.horizon == 0
        {
            return Err("replay requires an empty single-pool simulation at time zero, workloads and a positive horizon".into());
        }
        let mut ids = BTreeSet::new();
        for w in &self.workloads {
            if let Some(f) = &w.forecast
                && f.rates
                    .iter()
                    .any(|(n, rate)| *rate == 0 || !w.spec.supports_node_count(*n))
            {
                return Err(format!(
                    "invalid forecast rate or gang size for {}",
                    w.spec.id
                ));
            }
            if !ids.insert(&w.spec.id)
                || w.at > self.horizon
                || w.work == 0
                || w.spec
                    .node_counts()
                    .any(|n| w.rates.get(&n).copied().unwrap_or(0) == 0)
            {
                return Err(format!(
                    "invalid arrival, work, rates or duplicate ID for {}",
                    w.spec.id
                ));
            }
        }
        [
            Sizing::Fixed,
            Sizing::Fallback,
            Sizing::Adaptive,
            Sizing::CostAware,
        ]
        .into_iter()
        .map(|p| self.run(p))
        .collect()
    }

    fn next_event(&self, c: &Cluster, results: &BTreeMap<String, JobResult>) -> Timestamp {
        let mut next = self.horizon;
        let mut consider = |t| {
            if t > c.now {
                next = next.min(t);
            }
        };
        let queued = c.jobs.values().any(|j| j.state == JobState::Queued);
        for w in &self.workloads {
            consider(w.at);
            let Some(job) = c.jobs.get(&w.spec.id) else {
                continue;
            };
            if job.state != JobState::Running {
                continue;
            }
            let a = job.attempts.last().unwrap();
            if queued && job.spec.resumable {
                // Sliding-window fair-share ranks can cross between workload events.
                consider(c.now.saturating_add(1));
            }
            consider(a.started_at.saturating_add(job.spec.protected_seconds));
            if let Some(r) = a.pending_resize() {
                consider(
                    r.deadline
                        .min(r.requested_at.saturating_add(w.checkpoint_seconds)),
                );
            } else {
                let remaining = w.work - results[&w.spec.id].progress;
                let rate = w.rates[&(a.node_count())];
                let ready = w.ready_at(a).max(c.now);
                consider(ready);
                consider(ready.saturating_add(remaining.div_ceil(rate)));
            }
        }
        next
    }

    fn settle(
        &self,
        c: &mut Cluster,
        pool: &str,
        results: &mut BTreeMap<String, JobResult>,
    ) -> Result<(), String> {
        // Each admission and reply changes this timestamp's snapshot, including
        // zero-cost checkpoints. Settle it before advancing the application clocks.
        loop {
            let mut changed = false;
            for w in &self.workloads {
                let id = &w.spec.id;
                let Some(result) = results.get(id) else {
                    continue;
                };
                w.report_if_ready(c, result)?;
                if w.acknowledge_checkpoint(c, result.progress)? {
                    changed = true;
                }
            }
            match scheduler::simulate(c, pool).map(|d| d.action) {
                Some(Action::Start { preempt, .. }) => {
                    for victim in preempt {
                        results.get_mut(&victim).unwrap().preemptions += 1;
                    }
                }
                Some(Action::Resize { .. }) => {}
                _ if !changed => break,
                _ => {}
            }
        }
        Ok(())
    }

    fn run(&self, sizing: Sizing) -> Result<Run, String> {
        let mut c = self.inventory.clone();
        let pool = c.pools.keys().next().unwrap().clone();
        let mut results = BTreeMap::<String, JobResult>::new();
        let mut now = 0;
        let mut previous = 0;
        loop {
            for workload in &self.workloads {
                if let Some(result) = results.get_mut(&workload.spec.id) {
                    workload.account_interval(&c.jobs[&workload.spec.id], result, previous, now);
                }
            }
            c.advance(now)?;
            for w in &self.workloads {
                let id = &w.spec.id;
                if w.at == now {
                    c.submit(sizing.job_spec(&w.spec))?;
                    results.insert(id.clone(), JobResult::default());
                }
                if results
                    .get(id)
                    .is_some_and(|r| r.progress == w.work && r.finished_at.is_none())
                {
                    c.finish(id, false)?;
                    results.get_mut(id).unwrap().finished_at = Some(now);
                }
            }
            self.settle(&mut c, &pool, &mut results)?;
            if now == self.horizon
                || (results.len() == self.workloads.len()
                    && results.values().all(|r| r.finished_at.is_some()))
            {
                break;
            }
            previous = now;
            now = self.next_event(&c, &results);
        }
        c.validate()?;
        for (id, r) in &mut results {
            r.first_start = c.jobs[id].attempts.first().map(|a| a.started_at);
            r.gpu_seconds = c.jobs[id]
                .attempts
                .iter()
                .map(|a| a.gpu_seconds(0, c.now))
                .sum();
            r.attempts = c.jobs[id].attempts.len();
            for a in &c.jobs[id].attempts {
                for resize in &a.resizes {
                    if let Some(prediction) = &resize.estimate {
                        r.growth_estimates.push(GrowthResult {
                            attempt: a.number,
                            requested_at: resize.requested_at,
                            nodes: resize.request.nodes,
                            prediction: prediction.clone(),
                            observed_seconds_to_finish: r
                                .finished_at
                                .map(|t| t - resize.requested_at),
                        });
                    }
                }
            }
        }
        Ok(Run {
            policy: match sizing {
                Sizing::Fixed => "fixed",
                Sizing::Fallback => "fallback",
                Sizing::Adaptive => "adaptive",
                Sizing::CostAware => "cost_aware",
            },
            as_of: c.now,
            jobs: results,
            fair_share: scheduler::pool_usage(&c, &pool),
        })
    }
}
