//! Pure, deterministic policy. Simulator execution is deliberately separate from planning.
pub use crate::placement::resize_fits;
use crate::{
    model::*,
    placement::{AvailableGpus, largest_resize},
};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    pub job: String,
    pub reason: String,
    pub action: Action,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    Start {
        placement: Vec<Placement>,
        preempt: Vec<String>,
    },
    Resize {
        request: crate::resize::ResizeRequest,
        #[serde(skip_serializing_if = "Option::is_none")]
        estimate: Option<crate::forecast::Estimate>,
    },
    Wait,
}

fn path<'a>(cluster: &'a Cluster, id: &'a str) -> Vec<&'a str> {
    let mut result = vec![];
    let mut current = Some(id);
    while let Some(key) = current {
        result.push(key);
        current = cluster.accounts[key].parent.as_deref();
    }
    result.reverse();
    result
}

#[derive(Debug, Serialize)]
pub struct PoolUsage {
    pub since: Timestamp,
    pub as_of: Timestamp,
    /// Charged GPU-seconds, including each account's descendants.
    pub accounts: BTreeMap<String, u128>,
}

pub fn pool_usage(cluster: &Cluster, pool: &str) -> PoolUsage {
    let mut accounts: BTreeMap<_, _> = cluster
        .accounts
        .values()
        .filter(|a| a.pool == pool)
        .map(|a| (a.id.clone(), 0))
        .collect();
    let since = cluster.now.saturating_sub(cluster.lookback_seconds);
    for job in cluster
        .jobs
        .values()
        .filter(|j| j.spec.pool == pool && j.spec.protected_seconds > 0)
    {
        let amount: u128 = job
            .attempts
            .iter()
            .map(|attempt| attempt.gpu_seconds(since, cluster.now))
            .sum();
        for account in path(cluster, &job.spec.account) {
            *accounts.get_mut(account).expect("validated account") += amount;
        }
    }
    PoolUsage {
        since,
        as_of: cluster.now,
        accounts,
    }
}

fn compare(cluster: &Cluster, usage: &BTreeMap<String, u128>, a: &Job, b: &Job) -> Ordering {
    let funded = (a.spec.protected_seconds == 0).cmp(&(b.spec.protected_seconds == 0));
    if funded != Ordering::Equal {
        return funded;
    }
    for (left, right) in path(cluster, &a.spec.account)
        .iter()
        .zip(path(cluster, &b.spec.account))
    {
        if *left != right {
            // Compare usage/weight without floating point or division by zero.
            let l =
                usage.get(*left).copied().unwrap_or(0) * u128::from(cluster.accounts[right].weight);
            let r =
                usage.get(right).copied().unwrap_or(0) * u128::from(cluster.accounts[*left].weight);
            // Keep sibling subtrees contiguous even when their usage ties.
            // Falling back to job age here could create a non-transitive order
            // when jobs inside one subtree have different descendant ranks.
            return l.cmp(&r).then_with(|| left.cmp(&right));
        }
    }
    (a.submitted_at, &a.spec.id).cmp(&(b.submitted_at, &b.spec.id))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Reclaim {
    Starting,
    Stopping,
    NonResumable,
    Protected { until: Timestamp },
    UnfundedRequest,
    FairShare,
    Eligible,
}

fn reclaim(
    cluster: &Cluster,
    usage: &BTreeMap<String, u128>,
    requester: &Job,
    owner: &Job,
) -> Reclaim {
    if owner.state == JobState::Starting {
        return Reclaim::Starting;
    }
    if owner.state == JobState::Stopping {
        return Reclaim::Stopping;
    }
    if !owner.spec.resumable {
        return Reclaim::NonResumable;
    }
    let start = cluster
        .executions
        .get(&owner.spec.id)
        .and_then(|e| e.running_at)
        .unwrap_or_else(|| owner.attempts.last().unwrap().started_at);
    if cluster.now.saturating_sub(start) < owner.spec.protected_seconds {
        return Reclaim::Protected {
            until: start.saturating_add(owner.spec.protected_seconds),
        };
    }
    if requester.spec.protected_seconds == 0 {
        return Reclaim::UnfundedRequest;
    }
    if compare(cluster, usage, requester, owner) != Ordering::Less {
        return Reclaim::FairShare;
    }
    Reclaim::Eligible
}

/// Propose one opportunistic expansion at a time. A decline or timeout ends
/// automatic proposals for that attempt; an operator can still request a resize.
fn next_growth(cluster: &Cluster, pool: &str, usage: &BTreeMap<String, u128>) -> Option<Decision> {
    let mut candidates = Vec::new();
    for job in cluster.jobs.values().filter(|j| j.spec.pool == pool) {
        let Some(a) = job.attempts.last() else {
            continue;
        };
        if (job.state == JobState::Stopping && a.checkpoint().is_some())
            || a.pending_resize().is_some()
        {
            return None;
        }
        if matches!(
            job.spec.resize_policy,
            Some(crate::resize::Policy::GrowWhenIdle | crate::resize::Policy::CostAware)
        ) && job.state == JobState::Running
            && a.resizes.is_empty()
            && a.placement.len() < job.spec.nodes as usize
            && crate::resize::ready_to_resize(cluster, job, a).is_ok()
        {
            candidates.push(job);
        }
    }
    candidates.sort_by(|a, b| compare(cluster, usage, a, b));
    for job in candidates {
        let attempt = job.attempts.last().unwrap();
        let choice = if job.spec.resize_policy == Some(crate::resize::Policy::CostAware) {
            crate::forecast::assess(cluster, job, attempt)
                .recommendation()
                .map(|(n, estimate)| (n, Some(estimate)))
        } else {
            largest_resize(cluster, job)
                .filter(|&n| n as usize > attempt.placement.len())
                .map(|n| (n, None))
        };
        if let Some((nodes, estimate)) = choice {
            let request = crate::resize::ResizeRequest {
                id: "auto-grow".into(),
                attempt: attempt.number,
                nodes,
            };
            return Some(Decision {
                job: job.spec.id.clone(),
                reason: match &estimate {
                    Some(e) => format!(
                        "idle capacity fits {nodes} nodes; estimated time to finish falls from {}s to {}s including checkpoint and restore",
                        e.stay_seconds, e.resize_seconds
                    ),
                    None => format!(
                        "pool queue is empty and the declared {nodes}-node size fits on idle capacity"
                    ),
                },
                action: Action::Resize { request, estimate },
            });
        }
    }
    None
}

/// Plan ONE admission against a snapshot. Recompute after applying each decision.
/// A wait does not reserve resources; starvation-safe backfill is future work.
pub fn next(cluster: &Cluster, pool: &str) -> Option<Decision> {
    let usage = pool_usage(cluster, pool).accounts;
    let mut queued: Vec<_> = cluster
        .jobs
        .values()
        .filter(|j| j.state == JobState::Queued && j.spec.pool == pool)
        .collect();
    if queued.is_empty() {
        return next_growth(cluster, pool, &usage);
    }
    queued.sort_by(|a, b| compare(cluster, &usage, a, b));
    let available = AvailableGpus::for_pool(cluster, pool);
    for job in &queued {
        if let Some(decision) = plan_admission(cluster, job, &usage, &available) {
            return Some(decision);
        }
    }
    Some(Decision {
        job: queued[0].spec.id.clone(),
        reason: "insufficient compatible capacity; no eligible preemption plan".into(),
        action: Action::Wait,
    })
}

fn plan_admission(
    cluster: &Cluster,
    job: &Job,
    usage: &BTreeMap<String, u128>,
    available: &AvailableGpus<'_>,
) -> Option<Decision> {
    let target = job.restore_target();
    if let Some(placement) = available.place(&job.spec, target) {
        let reason = if let Some(n) = target {
            format!("restoring checkpoint at the requested {n}-node size")
        } else if placement.len() == job.spec.nodes as usize {
            "highest-ranked feasible job; free capacity".into()
        } else {
            format!(
                "preferred {}-node gang unavailable; starting declared {}-node gang on free capacity",
                job.spec.nodes,
                placement.len()
            )
        };
        return Some(Decision {
            job: job.spec.id.clone(),
            reason,
            action: Action::Start {
                placement,
                preempt: vec![],
            },
        });
    }
    let mut victims: Vec<_> = cluster
        .jobs
        .values()
        .filter(|other| {
            other.state == JobState::Running
                && other.spec.pool == job.spec.pool
                && reclaim(cluster, usage, job, other) == Reclaim::Eligible
        })
        .collect();
    victims.sort_by(|a, b| compare(cluster, usage, b, a));
    if victims.is_empty() {
        return None;
    }
    let mut reclaimed = available.clone();
    let mut preempt = vec![];
    for victim in victims {
        reclaimed.reclaim(victim.attempts.last().unwrap());
        preempt.push(victim.spec.id.clone());
        if let Some(placement) = reclaimed.place(&job.spec, target) {
            // A lower-ranked victim can be outside the final gang's placement.
            preempt.retain(|id| {
                cluster.jobs[id]
                    .attempts
                    .last()
                    .unwrap()
                    .overlaps(&placement)
            });
            return Some(Decision {
                job: job.spec.id.clone(),
                reason: "higher fair-share rank; victims are resumable and outside protection"
                    .into(),
                action: Action::Start { placement, preempt },
            });
        }
    }
    None
}

/// Instantaneous stop/start is ONLY a simulation assumption. A real executor must
/// checkpoint, confirm termination, and fence stale attempts before reuse.
pub fn simulate(cluster: &mut Cluster, pool: &str) -> Option<Decision> {
    if cluster.mode != Mode::Simulation {
        return None;
    }
    let decision = next(cluster, pool)?;
    match &decision.action {
        Action::Start { placement, preempt } => {
            for id in preempt {
                let victim = cluster.jobs.get_mut(id).unwrap();
                victim.attempts.last_mut().unwrap().release(cluster.now);
                victim.state = JobState::Queued;
            }
            let job = cluster.jobs.get_mut(&decision.job).unwrap();
            job.reserve_attempt(cluster.now, placement);
            job.state = JobState::Running;
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
    Some(decision)
}

/// A read-only explanation at the same logical time as the scheduler snapshot.
#[derive(Debug, Serialize)]
pub struct JobExplanation {
    pub as_of: Timestamp,
    pub state: JobState,
    pub queue_rank: Option<usize>,
    pub requested_nodes: Vec<u32>,
    pub placement_on_free_devices: Option<Vec<Placement>>,
    pub pool_decision: Option<Decision>,
    pub allocation_owners: BTreeMap<String, Reclaim>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub growth_forecast: Option<crate::forecast::Assessment>,
    pub nodes: BTreeMap<String, NodeAvailability>,
}

#[derive(Debug, Serialize)]
pub struct NodeAvailability {
    pub schedulable: bool,
    pub recent_heartbeat: Option<bool>,
    pub supports_checkpoint: Option<bool>,
    pub distributed_address: bool,
    /// Absent when the scheduler excludes the node.
    pub free_gpus: Option<usize>,
    pub largest_free_domain: Option<usize>,
}

pub fn explain(cluster: &Cluster, job: &Job) -> JobExplanation {
    let pool = &job.spec.pool;
    let available = AvailableGpus::for_pool(cluster, pool);
    let usage = pool_usage(cluster, pool);
    let queue_rank = (job.state == JobState::Queued).then(|| {
        1 + cluster
            .jobs
            .values()
            .filter(|other| {
                other.state == JobState::Queued
                    && other.spec.pool == *pool
                    && compare(cluster, &usage.accounts, other, job) == Ordering::Less
            })
            .count()
    });
    let managed = cluster.mode == Mode::Managed;
    let nodes = cluster
        .nodes
        .values()
        .filter(|n| n.pool == *pool)
        .map(|n| {
            let devices = available.devices(&n.id);
            let domain = devices.and_then(|free| {
                n.gpu_domains
                    .values()
                    .map(|d| d.iter().filter(|gpu| free.contains(gpu)).count())
                    .max()
            });
            (
                n.id.clone(),
                NodeAvailability {
                    schedulable: n.schedulable,
                    recent_heartbeat: managed.then(|| cluster.recent_heartbeat(&n.id)),
                    supports_checkpoint: managed.then(|| cluster.checkpoint_nodes.contains(&n.id)),
                    distributed_address: n.address.is_some(),
                    free_gpus: devices.map(<[u32]>::len),
                    largest_free_domain: domain,
                },
            )
        })
        .collect();
    let target = job.restore_target();
    let requested_nodes = job
        .spec
        .node_counts()
        .filter(|&n| target.is_none_or(|t| t == n))
        .collect();
    let allocation_owners = if queue_rank.is_some() {
        cluster
            .jobs
            .values()
            .filter(|owner| owner.spec.pool == *pool && owner.state.holds_allocation())
            .map(|owner| {
                (
                    owner.spec.id.clone(),
                    reclaim(cluster, &usage.accounts, job, owner),
                )
            })
            .collect()
    } else {
        BTreeMap::new()
    };
    JobExplanation {
        growth_forecast: job
            .attempts
            .last()
            .filter(|a| {
                job.state == JobState::Running
                    && a.resizes.is_empty()
                    && job.spec.resize_policy == Some(crate::resize::Policy::CostAware)
            })
            .map(|a| crate::forecast::assess(cluster, job, a)),
        allocation_owners,
        as_of: cluster.now,
        state: job.state.clone(),
        queue_rank,
        requested_nodes,
        placement_on_free_devices: queue_rank.and_then(|_| available.place(&job.spec, target)),
        pool_decision: next(cluster, pool),
        nodes,
    }
}
