#!/usr/bin/env python3
"""Submit a fixed-size PyTorch collective on a dedicated Soul pool."""
import argparse
from contextlib import closing
import http.client
import json
import os
from pathlib import Path
import re
import sys
import time
from uuid import uuid4


WORKER = Path(__file__).resolve().parents[1] / "examples/nccl_canary.py"
LAUNCH = """\
worker=$1
job=$2
if [ "$SOUL_NODE_COUNT" = 1 ]; then
    set -- --standalone
else
    set -- --nnodes="$SOUL_NODE_COUNT" --node-rank="$SOUL_NODE_RANK" \\
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --rdzv-id="$SOUL_RENDEZVOUS_ID"
fi
exec python -m torch.distributed.run "$@" --nproc-per-node="$SOUL_GPUS_PER_NODE" \\
    --max-restarts=0 --no-python python -c "$worker" "$job"
"""
TERMINAL = {"completed", "failed", "cancelled"}


class ApiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Controller:
    def __init__(self, port, token):
        self.port = port
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def request(self, method, path, body=None):
        # The controller and operator SSH forwards are both loopback-only.
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)) as conn:
            conn.request(method, path, json.dumps(body) if body is not None else None, self.headers)
            response = conn.getresponse()
            data = response.read()
            if response.status >= 300:
                detail = data[:256].decode(errors="replace")
                raise ApiError(response.status, f"{method} {path}: HTTP {response.status} {detail}")
            return json.loads(data) if data else None


def wait_for_exit(api, path, seconds, cancel=False):
    deadline = time.monotonic() + seconds
    last = "waiting for controller"
    while time.monotonic() < deadline:
        try:
            if cancel:
                api.request("POST", path + "/cancel")
                cancel = False
            job = api.request("GET", path)
            last = job["state"]
            if last in TERMINAL:
                return job
        except (OSError, http.client.HTTPException) as error:
            last = str(error)
        except ApiError as error:
            if error.status < 500:
                raise
            last = str(error)
        time.sleep(1)
    raise TimeoutError(f"{path} timed out ({last})")


def qualify(api, spec, timeout):
    controller = api.request("GET", "/v1/controller")
    if controller["mode"] != "managed" or not controller["controller_id"]:
        raise RuntimeError("a managed controller is required")
    job_id = spec["id"]
    path = f"/v1/jobs/{job_id}"
    print(f"Controller {controller['controller_id']}, job {job_id}", file=sys.stderr, flush=True)
    submitted = False
    try:
        api.request("POST", "/v1/jobs", spec)
        submitted = True
        job = wait_for_exit(api, path, timeout)
    except (KeyboardInterrupt, OSError, http.client.HTTPException, ApiError) as failure:
        if not submitted and isinstance(failure, ApiError) and failure.status < 500:
            raise
        # Submission may have committed even if its acknowledgement was lost.
        print(f"Requesting cancellation of {job_id}", file=sys.stderr)
        try:
            stopped = wait_for_exit(api, path, 90, cancel=True)
            print(f"{job_id} reached {stopped['state']}", file=sys.stderr)
        except (OSError, http.client.HTTPException, ApiError) as error:
            print(f"Cleanup unconfirmed for {job_id}. Inspect {path} before releasing nodes: {error}",
                  file=sys.stderr)
        raise

    attempts = job["attempts"]
    if job["state"] != "completed" or len(attempts) != 1:
        raise RuntimeError(f"{job_id} ended {job['state']} after {len(attempts)} attempts; inspect {path}")
    attempt = attempts[0]
    placement = attempt["placement"]
    if (attempt["ended_at"] is None or len({p["node"] for p in placement}) != spec["nodes"]
            or len(placement) != spec["nodes"]
            or any(len(p["gpus"]) != spec["gpus_per_node"]
                   or len(set(p["gpus"])) != len(p["gpus"]) for p in placement)):
        raise RuntimeError(f"unexpected final allocation for {job_id}")
    return {"job": job_id, "image": spec["image"], "controller_id": controller["controller_id"],
            "state": job["state"], "placement": placement,
            "queue_seconds": attempt["started_at"] - job["submitted_at"],
            "allocation_seconds": attempt["ended_at"] - attempt["started_at"],
            "receipt_node": placement[0]["node"],
            "receipt_container": f"soul-{controller['controller_id']}-{placement[0]['node']}-{job_id}-{attempt['number']}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--image", required=True, help="Cached CUDA PyTorch image reference pinned by digest")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--port", type=int, default=8080, help="Controller loopback port")
    parser.add_argument("--timeout", type=int, default=420, help="Queue and execution budget in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print the job without contacting a controller")
    args = parser.parse_args()
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", args.image):
        parser.error("supply an image pinned by @sha256 digest")
    if min(args.nodes, args.gpus_per_node, args.timeout) <= 0 or not 1 <= args.port <= 65535:
        parser.error("counts and timeout must be positive, port must be 1–65535")
    job = f"canary-{uuid4().hex}"
    spec = {"id": job, "pool": args.pool, "account": args.account, "image": args.image,
            "command": ["sh", "-eu", "-c", LAUNCH, "soul-canary", WORKER.read_text(), job],
            "nodes": args.nodes, "gpus_per_node": args.gpus_per_node,
            "protected_seconds": 0, "resumable": False}
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return
    token = os.environ.get("SOUL_TOKEN")
    if not token:
        parser.error("set SOUL_TOKEN for the managed controller")
    print(json.dumps(qualify(Controller(args.port, token), spec, args.timeout), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, http.client.HTTPException) as error:
        sys.exit(str(error))
