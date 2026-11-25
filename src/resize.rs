//! Cooperative checkpoint and restart of a gang at an explicitly supported size.
use crate::{
    execution::{self, Phase, StopReason},
    model::*,
    placement,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    Manual,
    GrowWhenIdle,
    CostAware,
}

pub const CHECKPOINT_TIMEOUT: u64 = 300;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ResizeRequest {
    pub id: String,
    pub attempt: u32,
    pub nodes: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Checkpoint {
    pub reference: String,
    pub progress: u64,
}

impl Checkpoint {
    fn valid(&self) -> bool {
        !self.reference.trim().is_empty()
            && self.reference.len() <= 4096
            && !self.reference.chars().any(char::is_control)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Reply {
    Ready { checkpoint: Checkpoint },
    Declined,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum Outcome {
    Pending,
    Ready { checkpoint: Checkpoint },
    Declined,
    Expired,
    Interrupted,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Resize {
    #[serde(flatten)]
    pub request: ResizeRequest,
    pub requested_at: Timestamp,
    pub deadline: Timestamp,
    pub updated_at: Timestamp,
    pub outcome: Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimate: Option<crate::forecast::Estimate>,
}

impl Attempt {
    pub(crate) fn pending_resize(&self) -> Option<&Resize> {
        self.resizes
            .last()
            .filter(|resize| resize.outcome == Outcome::Pending)
    }

    pub fn checkpoint(&self) -> Option<(&Resize, &Checkpoint)> {
        let resize = self.resizes.last()?;
        match &resize.outcome {
            Outcome::Ready { checkpoint } => Some((resize, checkpoint)),
            _ => None,
        }
    }

    pub fn close_pending_resize(&mut self, now: Timestamp, outcome: Outcome) {
        if let Some(r) = self.resizes.last_mut()
            && r.outcome == Outcome::Pending
        {
            r.outcome = outcome;
            r.updated_at = now;
        }
    }
}

pub fn expire(cluster: &mut Cluster) {
    for job in cluster
        .jobs
        .values_mut()
        .filter(|j| j.state == JobState::Running)
    {
        if let Some(attempt) = job.attempts.last_mut()
            && attempt
                .resizes
                .last()
                .is_some_and(|r| cluster.now >= r.deadline)
        {
            attempt.close_pending_resize(cluster.now, Outcome::Expired);
        }
    }
}

pub(crate) fn ready_to_resize(
    cluster: &Cluster,
    job: &Job,
    attempt: &Attempt,
) -> Result<(), String> {
    let running_at = if cluster.mode == Mode::Managed {
        let execution = &cluster.executions[&job.spec.id];
        if execution
            .members
            .values()
            .any(|m| m.phase != Phase::Running)
        {
            return Err("all gang members must be running".into());
        }
        execution.running_at.ok_or("gang has not started")?
    } else {
        attempt.started_at
    };
    if cluster.now.saturating_sub(running_at) < job.spec.protected_seconds {
        return Err("attempt is inside its protection window".into());
    }
    Ok(())
}

pub fn request(cluster: &mut Cluster, id: &str, input: ResizeRequest) -> Result<Resize, String> {
    record_request(cluster, id, input, None)
}

pub(crate) fn record_request(
    cluster: &mut Cluster,
    id: &str,
    input: ResizeRequest,
    estimate: Option<crate::forecast::Estimate>,
) -> Result<Resize, String> {
    if !valid_id(&input.id) {
        return Err("invalid resize request ID".into());
    }
    let job = cluster.jobs.get(id).ok_or("unknown job")?;
    let attempt = input
        .attempt
        .checked_sub(1)
        .and_then(|i| job.attempts.get(i as usize))
        .ok_or("unknown attempt")?;
    if let Some(prior) = attempt.resizes.iter().find(|r| r.request.id == input.id) {
        return if prior.request == input {
            Ok(prior.clone())
        } else {
            Err("resize ID already used with different parameters".into())
        };
    }
    if job.spec.resize_policy.is_none()
        || job.state != JobState::Running
        || Some(attempt.number) != job.attempts.last().map(|a| a.number)
    {
        return Err(
            "resize requires the current running attempt of a checkpoint-capable job".into(),
        );
    }
    if attempt.pending_resize().is_some() {
        return Err("another resize is pending".into());
    }
    if input.nodes as usize == attempt.placement.len() || !job.spec.supports_node_count(input.nodes)
    {
        return Err("target must be a different declared gang size".into());
    }
    ready_to_resize(cluster, job, attempt)?;
    if !placement::resize_fits(cluster, job, input.nodes) {
        return Err("target cannot fit on healthy capacity available to this job".into());
    }
    let resize = Resize {
        estimate,
        request: input,
        requested_at: cluster.now,
        deadline: cluster.now.saturating_add(CHECKPOINT_TIMEOUT),
        updated_at: cluster.now,
        outcome: Outcome::Pending,
    };
    cluster
        .jobs
        .get_mut(id)
        .unwrap()
        .attempts
        .last_mut()
        .unwrap()
        .resizes
        .push(resize.clone());
    Ok(resize)
}

pub fn reply(
    cluster: &mut Cluster,
    id: &str,
    number: u32,
    request_id: &str,
    reply: Reply,
) -> Result<Resize, String> {
    let outcome = match reply {
        Reply::Ready { checkpoint } => {
            if !checkpoint.valid() {
                return Err("invalid durable checkpoint reference".into());
            }
            Outcome::Ready { checkpoint }
        }
        Reply::Declined => Outcome::Declined,
    };
    let job = cluster.jobs.get_mut(id).ok_or("unknown job")?;
    let attempt = job
        .attempts
        .last_mut()
        .filter(|a| a.number == number)
        .ok_or("stale attempt")?;
    let resize = attempt
        .resizes
        .iter_mut()
        .find(|r| r.request.id == request_id)
        .ok_or("unknown resize")?;
    if resize.outcome == outcome {
        return Ok(resize.clone());
    }
    if job.state != JobState::Running
        || resize.outcome != Outcome::Pending
        || cluster.now >= resize.deadline
    {
        return Err("resize is no longer pending".into());
    }
    resize.outcome = outcome;
    resize.updated_at = cluster.now;
    let result = resize.clone();
    if matches!(result.outcome, Outcome::Ready { .. }) {
        if cluster.mode == Mode::Managed {
            execution::request_stop(cluster, id, StopReason::Resize)?;
        } else {
            attempt.release(cluster.now);
            job.state = JobState::Queued;
        }
    }
    Ok(result)
}

/// Verify replayed state before it can trigger a restart or a stop.
pub fn validate_history(job: &Job, now: Timestamp) -> Result<(), String> {
    for (index, attempt) in job.attempts.iter().enumerate() {
        if let Some(observation) = &attempt.profile {
            observation
                .profile
                .validate(&job.spec, attempt.node_count())?;
            if observation.received_at < attempt.started_at
                || observation.received_at > attempt.ended_at.unwrap_or(now)
            {
                return Err("invalid performance observation time".into());
            }
        }
        let mut ids = std::collections::BTreeSet::new();
        let mut previous = attempt.started_at;
        for (i, resize) in attempt.resizes.iter().enumerate() {
            let terminal_entry = i + 1 == attempt.resizes.len();
            let current = index + 1 == job.attempts.len();
            if job.spec.resize_policy.is_none()
                || !valid_id(&resize.request.id)
                || !ids.insert(&resize.request.id)
                || resize.request.attempt != attempt.number
                || !job.spec.supports_node_count(resize.request.nodes)
                || resize.request.nodes as usize == attempt.placement.len()
                || resize.requested_at < previous
                || resize.requested_at > resize.updated_at
                || resize.deadline != resize.requested_at.saturating_add(CHECKPOINT_TIMEOUT)
                || resize.updated_at > attempt.ended_at.unwrap_or(now)
            {
                return Err("invalid resize history".into());
            }
            match &resize.outcome {
                Outcome::Pending
                    if !terminal_entry
                        || !current
                        || job.state != JobState::Running
                        || resize.updated_at != resize.requested_at =>
                {
                    return Err("orphan checkpoint request".into());
                }
                Outcome::Ready { checkpoint }
                    if !terminal_entry
                        || !checkpoint.valid()
                        || resize.updated_at >= resize.deadline =>
                {
                    return Err("invalid ready checkpoint".into());
                }
                Outcome::Expired if resize.updated_at < resize.deadline => {
                    return Err("premature checkpoint expiry".into());
                }
                _ => {}
            }
            previous = resize.updated_at;
        }
        if let Some((resize, _)) = attempt.checkpoint()
            && let Some(next) = job.attempts.get(index + 1)
            && (next.placement.len() != resize.request.nodes as usize
                || attempt.ended_at.is_none_or(|end| end > next.started_at))
        {
            return Err("restored attempt differs from committed resize".into());
        }
    }
    Ok(())
}
