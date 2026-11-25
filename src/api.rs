//! HTTP shell. The managed server owns wall time; clients cannot advance it.
use crate::{execution, model::*, resize, scheduler, store::Store};
use axum::{
    Json, Router,
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::{
    io::Read,
    sync::{Arc, Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone)]
pub struct Service {
    pub store: Arc<Mutex<Store>>,
    pub token: Option<String>,
}
type ApiError = (StatusCode, String);
impl Service {
    fn lock_store(&self) -> Result<MutexGuard<'_, Store>, ApiError> {
        self.store.lock().map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "state lock unavailable".into(),
            )
        })
    }
}

fn bad(error: String) -> ApiError {
    (StatusCode::BAD_REQUEST, error)
}
fn now(cluster: &mut Cluster) -> Result<(), String> {
    if cluster.mode == Mode::Managed {
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();
        // Clock corrections must never rewind allocation intervals.
        cluster.advance(seconds.max(cluster.now))?;
    }
    Ok(())
}
pub fn tick(service: &Service) -> Result<(), String> {
    service
        .store
        .lock()
        .map_err(|_| "state lock unavailable")?
        .update(|cluster| {
            now(cluster)?;
            execution::reconcile(cluster);
            Ok(())
        })
}

async fn authorize(State(service): State<Service>, request: Request, next: Next) -> Response {
    if let Some(token) = &service.token {
        let expected = format!("Bearer {token}");
        if request
            .headers()
            .get("authorization")
            .and_then(|h| h.to_str().ok())
            != Some(expected.as_str())
        {
            return StatusCode::UNAUTHORIZED.into_response();
        }
    }
    next.run(request).await
}
async fn state(State(service): State<Service>) -> Result<Json<Cluster>, ApiError> {
    let mut snapshot = service.lock_store()?.state().clone();
    for execution in snapshot.executions.values_mut() {
        execution.runtime_token = None;
    }
    Ok(Json(snapshot))
}
#[derive(Serialize)]
struct ControllerInfo {
    mode: Mode,
    controller_id: Option<String>,
}
async fn controller(State(service): State<Service>) -> Result<Json<ControllerInfo>, ApiError> {
    let store = service.lock_store()?;
    let cluster = store.state();
    Ok(Json(ControllerInfo {
        mode: cluster.mode,
        controller_id: cluster.controller_id.clone(),
    }))
}
async fn job(
    State(service): State<Service>,
    Path(id): Path<String>,
) -> Result<Json<Job>, ApiError> {
    service
        .lock_store()?
        .state()
        .jobs
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "unknown job".into()))
}
async fn submit(
    State(service): State<Service>,
    Json(spec): Json<JobSpec>,
) -> Result<StatusCode, ApiError> {
    service
        .lock_store()?
        .update(|cluster| {
            now(cluster)?;
            cluster.submit(spec)?;
            execution::reconcile(cluster);
            Ok(())
        })
        .map_err(bad)?;
    Ok(StatusCode::ACCEPTED)
}
async fn cancel(
    State(service): State<Service>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    service
        .lock_store()?
        .update(|cluster| {
            now(cluster)?;
            if cluster.mode == Mode::Managed {
                execution::request_stop(cluster, &id, execution::StopReason::Cancel)
            } else {
                cluster.finish(&id, true)
            }
        })
        .map_err(bad)?;
    Ok(StatusCode::ACCEPTED)
}
async fn request_resize(
    State(service): State<Service>,
    Path(id): Path<String>,
    Json(input): Json<resize::ResizeRequest>,
) -> Result<Json<resize::Resize>, ApiError> {
    let result = service
        .lock_store()?
        .update(|c| {
            now(c)?;
            resize::request(c, &id, input)
        })
        .map_err(conflict)?;
    Ok(Json(result))
}

fn conflict(error: String) -> ApiError {
    (StatusCode::CONFLICT, error)
}

#[derive(Deserialize)]
struct ResizeReply {
    id: String,
    #[serde(flatten)]
    reply: resize::Reply,
}

fn authorize_runtime(
    c: &Cluster,
    job: &str,
    number: u32,
    headers: &HeaderMap,
) -> Result<(), ApiError> {
    let expected = c
        .executions
        .get(job)
        .filter(|e| e.attempt == number)
        .and_then(|e| e.runtime_token.as_deref());
    let token = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));
    if c.mode != Mode::Managed || expected.is_none() || token != expected {
        return Err((
            StatusCode::UNAUTHORIZED,
            "invalid attempt credential".into(),
        ));
    }
    Ok(())
}

async fn runtime_poll(
    State(service): State<Service>,
    Path((job, number)): Path<(String, u32)>,
    headers: HeaderMap,
) -> Result<Json<Option<resize::Resize>>, ApiError> {
    let store = service.lock_store()?;
    let c = store.state();
    authorize_runtime(c, &job, number, &headers)?;
    let pending = c.jobs[&job]
        .attempts
        .last()
        .and_then(Attempt::pending_resize)
        .cloned();
    Ok(Json(pending))
}

async fn runtime_reply(
    State(service): State<Service>,
    Path((job, number)): Path<(String, u32)>,
    headers: HeaderMap,
    Json(body): Json<ResizeReply>,
) -> Result<Json<resize::Resize>, ApiError> {
    let mut store = service.lock_store()?;
    authorize_runtime(store.state(), &job, number, &headers)?;
    store
        .update(|c| {
            now(c)?;
            resize::reply(c, &job, number, &body.id, body.reply)
        })
        .map(Json)
        .map_err(conflict)
}

async fn runtime_profile(
    State(service): State<Service>,
    Path((job, number)): Path<(String, u32)>,
    headers: HeaderMap,
    Json(profile): Json<crate::forecast::Profile>,
) -> Result<StatusCode, ApiError> {
    let mut store = service.lock_store()?;
    authorize_runtime(store.state(), &job, number, &headers)?;
    store
        .update(|c| {
            now(c)?;
            crate::forecast::report(c, &job, number, profile)
        })
        .map_err(conflict)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn simulation_profile(
    State(service): State<Service>,
    Path((job, number)): Path<(String, u32)>,
    Json(profile): Json<crate::forecast::Profile>,
) -> Result<StatusCode, ApiError> {
    service
        .lock_store()?
        .update(|c| {
            if c.mode != Mode::Simulation {
                return Err("simulation profile requires simulation mode".into());
            }
            crate::forecast::report(c, &job, number, profile)
        })
        .map_err(conflict)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn simulation_reply(
    State(service): State<Service>,
    Path((job, number)): Path<(String, u32)>,
    Json(body): Json<ResizeReply>,
) -> Result<Json<resize::Resize>, ApiError> {
    service
        .lock_store()?
        .update(|c| {
            if c.mode != Mode::Simulation {
                return Err("simulation reply requires simulation mode".into());
            }
            resize::reply(c, &job, number, &body.id, body.reply)
        })
        .map(Json)
        .map_err(conflict)
}

// Issue a credential only to the coordinator of the committed attempt. The store
// persists it before returning work so a lost poll or controller restart is safe.
fn issue_runtime_access(c: &mut Cluster, node: &str) -> Result<(), String> {
    for (id, e) in &mut c.executions {
        let job = &c.jobs[id];
        if job.spec.resize_policy.is_some()
            && e.runtime_token.is_none()
            && job.state == JobState::Starting
            && job.attempts.last().unwrap().placement[0].node == node
        {
            let mut bytes = [0u8; 32];
            std::fs::File::open("/dev/urandom")
                .and_then(|mut f| f.read_exact(&mut bytes))
                .map_err(|e| e.to_string())?;
            e.runtime_token = Some(bytes.iter().map(|b| format!("{b:02x}")).collect());
        }
    }
    Ok(())
}

async fn explain(
    State(service): State<Service>,
    Path(id): Path<String>,
) -> Result<Json<scheduler::JobExplanation>, ApiError> {
    let store = service.lock_store()?;
    let job = store
        .state()
        .jobs
        .get(&id)
        .ok_or((StatusCode::NOT_FOUND, "unknown job".into()))?;
    Ok(Json(scheduler::explain(store.state(), job)))
}

async fn plan(
    State(service): State<Service>,
    Path(pool): Path<String>,
) -> Result<Json<Option<scheduler::Decision>>, ApiError> {
    let store = service.lock_store()?;
    if !store.state().pools.contains_key(&pool) {
        return Err(bad("unknown pool".into()));
    }
    Ok(Json(scheduler::next(store.state(), &pool)))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SchedulingUpdate {
    schedulable: bool,
}

async fn node_scheduling(
    State(service): State<Service>,
    Path(node): Path<String>,
) -> Result<Json<NodeScheduling>, ApiError> {
    service
        .lock_store()?
        .state()
        .node_scheduling(&node)
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "unknown node".into()))
}

async fn set_node_scheduling(
    State(service): State<Service>,
    Path(node): Path<String>,
    Json(input): Json<SchedulingUpdate>,
) -> Result<Json<NodeScheduling>, ApiError> {
    let mut store = service.lock_store()?;
    if !store.state().nodes.contains_key(&node) {
        return Err((StatusCode::NOT_FOUND, "unknown node".into()));
    }
    store
        .update(|cluster| {
            cluster.nodes.get_mut(&node).unwrap().schedulable = input.schedulable;
            Ok(())
        })
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(store.state().node_scheduling(&node).unwrap()))
}

async fn usage(
    State(service): State<Service>,
    Path(pool): Path<String>,
) -> Result<Json<scheduler::PoolUsage>, ApiError> {
    let store = service.lock_store()?;
    if !store.state().pools.contains_key(&pool) {
        return Err((StatusCode::NOT_FOUND, "unknown pool".into()));
    }
    Ok(Json(scheduler::pool_usage(store.state(), &pool)))
}
#[derive(Deserialize)]
struct Step {
    now: u64,
    pool: String,
}
async fn step(
    State(service): State<Service>,
    Json(step): Json<Step>,
) -> Result<Json<Option<scheduler::Decision>>, ApiError> {
    let result = service
        .lock_store()?
        .update(|cluster| {
            if cluster.mode != Mode::Simulation {
                return Err("simulation endpoint unavailable in managed mode".into());
            }
            if !cluster.pools.contains_key(&step.pool) {
                return Err("unknown pool".into());
            }
            cluster.advance(step.now)?;
            Ok(scheduler::simulate(cluster, &step.pool))
        })
        .map_err(bad)?;
    Ok(Json(result))
}
async fn complete(
    State(service): State<Service>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    service
        .lock_store()?
        .update(|cluster| cluster.finish(&id, false))
        .map_err(bad)?;
    Ok(StatusCode::NO_CONTENT)
}
async fn work(
    State(service): State<Service>,
    Path(node): Path<String>,
    headers: HeaderMap,
) -> Result<Json<execution::Work>, ApiError> {
    let result = service
        .lock_store()?
        .update(|cluster| {
            if cluster.mode != Mode::Managed {
                return Err("node protocol requires managed mode".into());
            }
            now(cluster)?;
            execution::heartbeat(cluster, &node)?;
            let capable = headers
                .get("x-soul-checkpoint-version")
                .is_some_and(|v| v == "1");
            if capable {
                cluster.checkpoint_nodes.insert(node.clone());
            } else {
                cluster.checkpoint_nodes.remove(&node);
            }
            execution::reconcile(cluster);
            if capable {
                issue_runtime_access(cluster, &node)?;
            }
            execution::work(cluster, &node)
        })
        .map_err(bad)?;
    Ok(Json(result))
}
async fn report(
    State(service): State<Service>,
    Path(node): Path<String>,
    Json(report): Json<execution::Report>,
) -> Result<StatusCode, ApiError> {
    service
        .lock_store()?
        .update(|cluster| {
            now(cluster)?;
            execution::report(cluster, &node, report)?;
            execution::heartbeat(cluster, &node)?;
            execution::reconcile(cluster);
            Ok(())
        })
        .map_err(bad)?;
    Ok(StatusCode::NO_CONTENT)
}
pub fn router(service: Service) -> Router {
    let controller = Router::new()
        .route("/v1/controller", get(controller))
        .route("/v1/state", get(state))
        .route("/v1/jobs", post(submit))
        .route("/v1/jobs/{id}", get(job))
        .route("/v1/jobs/{id}/explain", get(explain))
        .route(
            "/v1/simulation/jobs/{id}/{attempt}/profile",
            post(simulation_profile),
        )
        .route("/v1/jobs/{id}/resize", post(request_resize))
        .route(
            "/v1/simulation/jobs/{id}/{attempt}/checkpoint",
            post(simulation_reply),
        )
        .route("/v1/jobs/{id}/cancel", post(cancel))
        .route("/v1/pools/{pool}/plan", get(plan))
        .route("/v1/pools/{pool}/usage", get(usage))
        .route("/v1/simulation/step", post(step))
        .route("/v1/simulation/jobs/{id}/complete", post(complete))
        .route("/v1/nodes/{node}/work", post(work))
        .route(
            "/v1/nodes/{node}/scheduling",
            get(node_scheduling).put(set_node_scheduling),
        )
        .route("/v1/nodes/{node}/report", post(report))
        .layer(middleware::from_fn_with_state(service.clone(), authorize));
    // Runtime requests authenticate against their allocation attempt under the store lock.
    let runtime = Router::new()
        .route(
            "/v1/runtime/{id}/{attempt}",
            get(runtime_poll).post(runtime_reply),
        )
        .route("/v1/runtime/{id}/{attempt}/profile", post(runtime_profile));
    controller.merge(runtime).with_state(service)
}
