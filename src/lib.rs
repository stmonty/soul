//! GPU gang scheduling, allocation accounting and managed execution for Soul.
//!
//! The [`model`] module describes pools, nodes, accounts, jobs and attempts.
//! [`scheduler::next`] proposes a decision on a cluster snapshot. [`execution`]
//! reconciles committed allocations with node observations. [`replay::Trace`]
//! compares policies against a reproducible workload.
//!
//! For example, after running `soul example inventory > inventory.json`,
//! an application can load a snapshot and inspect the planner's next decision.
//!
//! ```no_run
//! use std::fs::File;
//! use soul::{model::Cluster, scheduler};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cluster: Cluster = serde_json::from_reader(File::open("inventory.json")?)?;
//! cluster.validate()?;
//! if let Some(decision) = scheduler::next(&cluster, "b300") {
//!     println!("{}", decision.reason);
//! }
//! # Ok(())
//! # }
//! ```
pub mod api;
pub mod execution;
pub mod forecast;
pub mod model;
pub mod node;
mod placement;
pub mod replay;
pub mod resize;
pub mod scheduler;
pub mod store;
