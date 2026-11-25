use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type Timestamp = u64; // Seconds supplied by the shell; logical in simulation, Unix in service.

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Simulation,
    Managed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pool {
    pub id: String,
    pub gpu_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    /// Reachable training-network hostname/IP, not the controller tunnel address.
    #[serde(default)]
    pub address: Option<String>,
    pub id: String,
    pub pool: String,
    pub gpus: u32,
    pub schedulable: bool,
    /// Disjoint node-local interconnect domains from operator inventory.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gpu_domains: BTreeMap<String, Vec<u32>>,
}

impl Node {
    pub fn can_host(&self, spec: &JobSpec) -> bool {
        self.pool == spec.pool
            && self.gpus >= spec.gpus_per_node
            && (!spec.same_gpu_domain
                || self
                    .gpu_domains
                    .values()
                    .any(|domain| domain.len() >= spec.gpus_per_node as usize))
    }

    /// Pick the smallest free domain that fits a locality-constrained group.
    pub fn select_devices(&self, free: &[u32], spec: &JobSpec) -> Option<Vec<u32>> {
        let count = spec.gpus_per_node as usize;
        if free.len() < count {
            return None;
        }
        if self.gpu_domains.is_empty() {
            return (!spec.same_gpu_domain).then(|| free[..count].to_vec());
        }
        let free_set: BTreeSet<_> = free.iter().copied().collect();
        let mut grouped = BTreeSet::new();
        let mut groups = Vec::new();
        for domain in self.gpu_domains.values() {
            let devices: Vec<_> = domain
                .iter()
                .copied()
                .filter(|gpu| free_set.contains(gpu))
                .collect();
            if !devices.is_empty() {
                grouped.extend(devices.iter().copied());
                groups.push((devices.len() == domain.len(), devices));
            }
        }
        // Stable ties use inventory domain name and the operator's device order.
        if spec.same_gpu_domain {
            let (_, mut devices) = groups
                .into_iter()
                .filter(|(_, g)| g.len() >= count)
                .min_by_key(|(_, g)| g.len())?;
            devices.truncate(count);
            return Some(devices);
        }
        groups.sort_by_key(|(intact, devices)| (*intact, devices.len()));
        let ungrouped = free.iter().copied().filter(|gpu| !grouped.contains(gpu));
        Some(
            ungrouped
                .chain(groups.into_iter().flat_map(|(_, g)| g))
                .take(count)
                .collect(),
        )
    }
}

/// Weights are relative to siblings. Accounts and usage are pool-specific.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Account {
    pub id: String,
    pub pool: String,
    pub parent: Option<String>,
    pub weight: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobSpec {
    /// Client-generated identity makes resubmission idempotent.
    pub id: String,
    pub pool: String,
    pub account: String,
    pub image: String,
    pub command: Vec<String>,
    pub nodes: u32,
    /// Smaller gang sizes this command supports, in descending preference order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallback_nodes: Vec<u32>,
    pub gpus_per_node: u32,
    /// Zero means opportunistic: uncharged, with no right to preempt other jobs.
    pub protected_seconds: u64,
    /// Application supports restart from its own durable checkpoint.
    pub resumable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resize_policy: Option<crate::resize::Policy>,
    #[serde(default)]
    pub same_gpu_domain: bool,
}

impl JobSpec {
    pub fn node_counts(&self) -> impl Iterator<Item = u32> + '_ {
        std::iter::once(self.nodes).chain(self.fallback_nodes.iter().copied())
    }

    pub fn supports_node_count(&self, nodes: u32) -> bool {
        self.nodes == nodes || self.fallback_nodes.contains(&nodes)
    }

    pub fn validate_shape(&self) -> Result<(), String> {
        if self.nodes == 0 || self.gpus_per_node == 0 {
            return Err("node and GPU counts must be positive".into());
        }
        let mut previous = self.nodes;
        for &count in &self.fallback_nodes {
            if count == 0 || count >= previous {
                return Err(
                    "fallback_nodes must contain positive, strictly decreasing sizes below nodes"
                        .into(),
                );
            }
            previous = count;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Starting,
    Running,
    Stopping,
    Completed,
    Cancelled,
    Failed,
}

impl JobState {
    pub(crate) fn holds_allocation(&self) -> bool {
        matches!(self, Self::Starting | Self::Running | Self::Stopping)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Placement {
    pub node: String,
    pub gpus: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub struct NodeScheduling {
    pub schedulable: bool,
    pub allocations: Vec<NodeAllocation>,
}

#[derive(Debug, Serialize)]
pub struct NodeAllocation {
    pub job: String,
    pub attempt: u32,
    pub gpus: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    pub number: u32,
    pub started_at: Timestamp,
    pub ended_at: Option<Timestamp>,
    pub placement: Vec<Placement>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resizes: Vec<crate::resize::Resize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<crate::forecast::Observation>,
}

impl Attempt {
    pub(crate) fn node_count(&self) -> u32 {
        self.placement.len() as u32
    }

    pub(crate) fn release(&mut self, now: Timestamp) {
        self.close_pending_resize(now, crate::resize::Outcome::Interrupted);
        self.ended_at.get_or_insert(now);
    }

    /// Preempt this gang only if the proposed placement needs one of its devices.
    pub fn overlaps(&self, placement: &[Placement]) -> bool {
        self.placement.iter().any(|held| {
            placement.iter().any(|requested| {
                held.node == requested.node
                    && requested.gpus.iter().any(|gpu| held.gpus.contains(gpu))
            })
        })
    }

    /// GPU-seconds of committed ownership overlapping [since, until).
    pub fn gpu_seconds(&self, since: Timestamp, until: Timestamp) -> u128 {
        let end = self.ended_at.unwrap_or(until).min(until);
        let seconds = end.saturating_sub(self.started_at.max(since));
        let devices: u128 = self.placement.iter().map(|p| p.gpus.len() as u128).sum();
        u128::from(seconds) * devices
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub spec: JobSpec,
    pub submitted_at: Timestamp,
    pub state: JobState,
    pub attempts: Vec<Attempt>,
}

impl Job {
    /// Open the ownership interval before any member can start.
    pub(crate) fn reserve_attempt(&mut self, now: Timestamp, placement: &[Placement]) -> u32 {
        let number = self.attempts.len() as u32 + 1;
        self.attempts.push(Attempt {
            number,
            started_at: now,
            ended_at: None,
            placement: placement.to_vec(),
            resizes: Vec::new(),
            profile: None,
        });
        self.state = JobState::Starting;
        number
    }

    pub(crate) fn restore_target(&self) -> Option<u32> {
        self.attempts
            .last()?
            .checkpoint()
            .map(|(resize, _)| resize.request.nodes)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cluster {
    #[serde(default)]
    pub controller_id: Option<String>,
    #[serde(default)]
    pub mode: Mode,
    #[serde(default)]
    pub executions: BTreeMap<String, crate::execution::Execution>,
    #[serde(default)]
    pub heartbeats: BTreeMap<String, Timestamp>,
    #[serde(default)]
    pub checkpoint_nodes: BTreeSet<String>,
    pub now: Timestamp,
    pub lookback_seconds: u64,
    pub max_protected_seconds: u64,
    pub pools: BTreeMap<String, Pool>,
    pub nodes: BTreeMap<String, Node>,
    pub accounts: BTreeMap<String, Account>,
    pub jobs: BTreeMap<String, Job>,
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            controller_id: None,
            mode: Mode::Simulation,
            executions: BTreeMap::new(),
            heartbeats: BTreeMap::new(),
            checkpoint_nodes: BTreeSet::new(),
            now: 0,
            lookback_seconds: 7 * 24 * 3600,
            max_protected_seconds: 8 * 3600,
            pools: BTreeMap::new(),
            nodes: BTreeMap::new(),
            accounts: BTreeMap::new(),
            jobs: BTreeMap::new(),
        }
    }
}

impl Cluster {
    pub(crate) fn active_attempts(&self, pool: &str) -> impl Iterator<Item = (&Job, &Attempt)> {
        self.jobs
            .values()
            .filter(move |job| job.spec.pool == pool && job.state.holds_allocation())
            .map(|job| {
                (
                    job,
                    job.attempts.last().expect("active job owns an allocation"),
                )
            })
    }

    pub fn node_scheduling(&self, id: &str) -> Option<NodeScheduling> {
        let node = self.nodes.get(id)?;
        let allocations = self
            .active_attempts(&node.pool)
            .filter_map(|(job, attempt)| {
                let placement = attempt.placement.iter().find(|p| p.node == id)?;
                Some(NodeAllocation {
                    job: job.spec.id.clone(),
                    attempt: attempt.number,
                    gpus: placement.gpus.clone(),
                })
            })
            .collect();
        Some(NodeScheduling {
            schedulable: node.schedulable,
            allocations,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.lookback_seconds == 0 {
            return Err("lookback must be positive".into());
        }
        for (id, pool) in &self.pools {
            if id.is_empty() || id != &pool.id {
                return Err("invalid pool identity".into());
            }
        }
        for (id, node) in &self.nodes {
            let mut assigned = BTreeSet::new();
            for (name, devices) in &node.gpu_domains {
                if !valid_id(name)
                    || devices.is_empty()
                    || devices
                        .iter()
                        .any(|gpu| *gpu >= node.gpus || !assigned.insert(*gpu))
                {
                    return Err(format!("invalid GPU domain {name} on {id}"));
                }
            }
            if node.address.as_ref().is_some_and(|address| {
                address.is_empty()
                    || address.len() > 253
                    || !address
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b".-:".contains(&b))
            }) {
                return Err(format!("invalid training address for {id}"));
            }
            if !valid_id(id)
                || id != &node.id
                || node.gpus == 0
                || !self.pools.contains_key(&node.pool)
            {
                return Err(format!("invalid node {id}"));
            }
        }
        for (id, account) in &self.accounts {
            if id.is_empty()
                || id != &account.id
                || account.weight == 0
                || !self.pools.contains_key(&account.pool)
            {
                return Err(format!("invalid account {id}"));
            }
            let mut seen = BTreeSet::new();
            let mut current = Some(id.as_str());
            while let Some(key) = current {
                if !seen.insert(key) {
                    return Err("account cycle".into());
                }
                let ancestor = self.accounts.get(key).ok_or("unknown parent")?;
                if ancestor.pool != account.pool {
                    return Err("cross-pool account parent".into());
                }
                current = ancestor.parent.as_deref();
            }
        }
        crate::execution::validate(self)?;
        Ok(())
    }

    pub fn submit(&mut self, spec: JobSpec) -> Result<(), String> {
        if let Some(existing) = self.jobs.get(&spec.id) {
            if existing.spec == spec {
                return Ok(());
            }
            return Err("job ID already exists with a different specification".into());
        }
        let account = self.accounts.get(&spec.account).ok_or("unknown account")?;
        if account.pool != spec.pool {
            return Err("account belongs to a different pool".into());
        }
        if self
            .accounts
            .values()
            .any(|a| a.parent.as_deref() == Some(&spec.account))
        {
            return Err("submit to a leaf account".into());
        }
        if !valid_id(&spec.id)
            || spec.image.is_empty()
            || spec.image.len() > 512
            || spec.image.starts_with('-')
            || spec.image.chars().any(char::is_whitespace)
            || spec.image.contains('\0')
            || spec.command.is_empty()
            || spec.command.first().is_some_and(String::is_empty)
            || spec.command.len() > 256
            || spec.command.iter().map(String::len).sum::<usize>() > 65536
            || spec.command.iter().any(|argument| argument.contains('\0'))
            || spec.protected_seconds > self.max_protected_seconds
        {
            return Err("invalid job specification".into());
        }
        spec.validate_shape()?;
        let smallest = spec.fallback_nodes.last().copied().unwrap_or(spec.nodes);
        let eligible = self
            .nodes
            .values()
            .filter(|n| {
                n.can_host(&spec)
                    && (self.mode != Mode::Managed || smallest == 1 || n.address.is_some())
            })
            .count();
        if eligible < smallest as usize {
            return Err("job exceeds pool topology".into());
        }
        self.jobs.insert(
            spec.id.clone(),
            Job {
                spec,
                submitted_at: self.now,
                state: JobState::Queued,
                attempts: vec![],
            },
        );
        Ok(())
    }

    pub(crate) fn recent_heartbeat(&self, node: &str) -> bool {
        self.heartbeats.get(node).is_some_and(|last| {
            self.now.saturating_sub(*last) <= crate::execution::HEARTBEAT_TIMEOUT
        })
    }

    pub fn advance(&mut self, now: Timestamp) -> Result<(), String> {
        if now < self.now {
            return Err("time cannot go backwards".into());
        }
        self.now = now;
        crate::resize::expire(self);
        Ok(())
    }

    pub fn finish(&mut self, id: &str, cancelled: bool) -> Result<(), String> {
        if self.mode != Mode::Simulation {
            return Err("managed execution requires node reports or a stop request".into());
        }
        let job = self.jobs.get_mut(id).ok_or("unknown job")?;
        if matches!(job.state, JobState::Completed | JobState::Cancelled) {
            return Ok(());
        }
        if !cancelled && job.state != JobState::Running {
            return Err("job is not running".into());
        }
        if let Some(attempt) = job.attempts.last_mut()
            && attempt.ended_at.is_none()
        {
            attempt.release(self.now);
        }
        job.state = if cancelled {
            JobState::Cancelled
        } else {
            JobState::Completed
        };
        Ok(())
    }
}

/// IDs are also used in HTTP paths and runtime labels.
pub fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
}
