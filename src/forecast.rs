//! Attempt-scoped application estimates for cost-aware growth on idle devices.
use crate::{
    model::*,
    placement,
    resize::{CHECKPOINT_TIMEOUT, Policy},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Profile {
    pub sequence: u64,
    pub valid_for_seconds: u64,
    pub remaining_work: u64,
    /// Useful work units per second, comparable across this job's launch sizes.
    pub rates: BTreeMap<u32, u64>,
    pub checkpoint_seconds: u64,
    pub restore_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub received_at: Timestamp,
    pub profile: Profile,
}

impl Profile {
    pub(crate) fn validate(&self, spec: &JobSpec, current: u32) -> Result<(), String> {
        if spec.resize_policy != Some(Policy::CostAware)
            || self.valid_for_seconds == 0
            || !self.rates.contains_key(&current)
            || self
                .rates
                .iter()
                .any(|(n, rate)| *rate == 0 || !spec.supports_node_count(*n))
        {
            return Err("a cost-aware profile requires a positive lifetime and positive rates for the current size and any profiled alternatives; sizes must be declared".into());
        }
        Ok(())
    }
}

pub fn report(c: &mut Cluster, id: &str, number: u32, profile: Profile) -> Result<(), String> {
    let job = c.jobs.get_mut(id).ok_or("unknown job")?;
    let a = job
        .attempts
        .last_mut()
        .filter(|a| a.number == number)
        .ok_or("stale profile attempt")?;
    if let Some(prior) = &a.profile {
        if prior.profile == profile {
            return Ok(());
        }
        if profile.sequence <= prior.profile.sequence {
            return Err("profile sequence must advance".into());
        }
    }
    if job.state != JobState::Running || !a.resizes.is_empty() {
        return Err("profiles require a running attempt before a resize request".into());
    }
    profile.validate(&job.spec, a.node_count())?;
    a.profile = Some(Observation {
        received_at: c.now,
        profile,
    });
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Estimate {
    pub sequence: u64,
    pub remaining_work: u64,
    pub stay_seconds: u64,
    pub resize_seconds: u64,
}

#[derive(Debug, Serialize)]
pub struct Alternative {
    pub nodes: u32,
    pub fits: bool,
    /// Includes checkpoint and restore. None means the application supplied no rate.
    pub seconds: Option<u128>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Assessment {
    AwaitingProfile,
    Expired {
        sequence: u64,
    },
    CheckpointExceedsDeadline {
        sequence: u64,
    },
    Compared {
        sequence: u64,
        remaining_work: u64,
        stay_seconds: u64,
        alternatives: Vec<Alternative>,
        recommended_nodes: Option<u32>,
    },
}

impl Assessment {
    pub fn recommendation(&self) -> Option<(u32, Estimate)> {
        let Self::Compared {
            sequence,
            remaining_work,
            stay_seconds,
            alternatives,
            recommended_nodes: Some(nodes),
        } = self
        else {
            return None;
        };
        let seconds = alternatives.iter().find(|a| a.nodes == *nodes)?.seconds?;
        // A recommended resize strictly beats stay_seconds, which fits in u64.
        Some((
            *nodes,
            Estimate {
                sequence: *sequence,
                remaining_work: *remaining_work,
                stay_seconds: *stay_seconds,
                resize_seconds: seconds as u64,
            },
        ))
    }
}

pub fn assess(c: &Cluster, job: &Job, a: &Attempt) -> Assessment {
    let Some(observation) = &a.profile else {
        return Assessment::AwaitingProfile;
    };
    let p = &observation.profile;
    let age = c.now.saturating_sub(observation.received_at);
    if age >= p.valid_for_seconds {
        return Assessment::Expired {
            sequence: p.sequence,
        };
    }
    if p.checkpoint_seconds >= CHECKPOINT_TIMEOUT {
        return Assessment::CheckpointExceedsDeadline {
            sequence: p.sequence,
        };
    }
    let nodes = a.node_count();
    let rate = p.rates[&nodes];
    let remaining_work = p.remaining_work.saturating_sub(age.saturating_mul(rate));
    let stay_seconds = remaining_work.div_ceil(rate);
    let capacity = placement::largest_resize(c, job).unwrap_or(0);
    let alternatives: Vec<_> = job
        .spec
        .node_counts()
        .take_while(|&n| n > nodes)
        .map(|n| {
            let seconds = p.rates.get(&n).map(|r| {
                u128::from(remaining_work.div_ceil(*r))
                    + u128::from(p.checkpoint_seconds)
                    + u128::from(p.restore_seconds)
            });
            Alternative {
                nodes: n,
                fits: n <= capacity,
                seconds,
            }
        })
        .collect();
    let recommended_nodes = alternatives
        .iter()
        .filter(|a| a.fits)
        .filter_map(|a| Some((a.seconds?, a.nodes)))
        .filter(|(seconds, _)| *seconds < u128::from(stay_seconds))
        .min()
        .map(|(_, n)| n);
    Assessment::Compared {
        sequence: p.sequence,
        remaining_work,
        stay_seconds,
        alternatives,
        recommended_nodes,
    }
}
