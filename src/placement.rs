//! Physical placement against a snapshot of one pool's allocatable devices.
use crate::model::*;
use std::collections::BTreeMap;

#[derive(Clone)]
pub(crate) struct AvailableGpus<'a> {
    cluster: &'a Cluster,
    free: BTreeMap<String, Vec<u32>>,
}

impl<'a> AvailableGpus<'a> {
    pub(crate) fn for_pool(cluster: &'a Cluster, pool: &str) -> Self {
        let mut free: BTreeMap<_, _> = cluster
            .nodes
            .values()
            .filter(|node| {
                node.pool == pool
                    && node.schedulable
                    && (cluster.mode == Mode::Simulation || cluster.recent_heartbeat(&node.id))
            })
            .map(|node| (node.id.clone(), (0..node.gpus).collect::<Vec<_>>()))
            .collect();
        for (_, attempt) in cluster.active_attempts(pool) {
            for held in &attempt.placement {
                if let Some(devices) = free.get_mut(&held.node) {
                    devices.retain(|gpu| !held.gpus.contains(gpu));
                }
            }
        }
        Self { cluster, free }
    }

    pub(crate) fn devices(&self, node: &str) -> Option<&[u32]> {
        self.free.get(node).map(Vec::as_slice)
    }

    /// Add back only devices on nodes still eligible for placement.
    pub(crate) fn reclaim(&mut self, attempt: &Attempt) {
        for held in &attempt.placement {
            if let Some(devices) = self.free.get_mut(&held.node) {
                devices.extend(&held.gpus);
                devices.sort_unstable();
            }
        }
    }

    pub(crate) fn place(&self, spec: &JobSpec, target: Option<u32>) -> Option<Vec<Placement>> {
        let eligible = |id: &str| {
            self.cluster.mode == Mode::Simulation
                || spec.resize_policy.is_none()
                || self.cluster.checkpoint_nodes.contains(id)
        };
        let mut nodes: Vec<_> = self
            .free
            .iter()
            .filter(|(id, _)| eligible(id))
            .filter_map(|(id, free)| {
                let selected = self.cluster.nodes[id].select_devices(free, spec)?;
                Some((id, free.len(), selected))
            })
            .collect();
        nodes.sort_by_key(|(id, free, _)| (*free, *id));
        let has_address = |id: &str| {
            self.cluster.mode == Mode::Simulation || self.cluster.nodes[id].address.is_some()
        };
        let distributed_nodes = nodes.iter().filter(|(id, _, _)| has_address(id)).count();
        let count = spec
            .node_counts()
            .filter(|&n| target.is_none_or(|t| n == t))
            .find(|&n| {
                if n == 1 {
                    !nodes.is_empty()
                } else {
                    n as usize <= distributed_nodes
                }
            })?;
        Some(
            nodes
                .into_iter()
                .filter(|(id, _, _)| count == 1 || has_address(id))
                .take(count as usize)
                .map(|(node, _, gpus)| Placement {
                    node: node.clone(),
                    gpus,
                })
                .collect(),
        )
    }
}

// All sizes request the same GPUs per node. Every smaller declared size fits
// wherever the largest feasible gang can run.
pub(crate) fn largest_resize(cluster: &Cluster, job: &Job) -> Option<u32> {
    let attempt = job.attempts.last()?;
    let mut available = AvailableGpus::for_pool(cluster, &job.spec.pool);
    available.reclaim(attempt);
    available
        .place(&job.spec, None)
        .map(|placement| placement.len() as u32)
}

pub fn resize_fits(cluster: &Cluster, job: &Job, target: u32) -> bool {
    job.spec.supports_node_count(target)
        && largest_resize(cluster, job).is_some_and(|n| n >= target)
}
