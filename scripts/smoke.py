#!/usr/bin/env python3
"""Actual Docker execution, cancellation, and controller/node restart smoke.
Uses one synthetic GPU and --cpu-test; no GPU or CUDA validation is claimed.
Requires cargo build --bins and a local ubuntu:24.04 Docker image.
"""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("SOUL_BIN_DIR", ROOT / "target/debug")).resolve()


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, SOUL_TOKEN=secrets.token_hex(24))
    processes = []
    containers = set()
    with tempfile.TemporaryDirectory(prefix="soul-smoke-") as directory:
        directory = Path(directory)
        logs = open(directory / "service.log", "w+")
        controller_cmd = [str(BIN / "soul"), "serve", str(directory / "controller.sqlite"), str(ROOT / "examples/small.json"), f"127.0.0.1:{port}"]
        node_cmd = [str(BIN / "soul-node"), base, "node-1", str(directory / "node.sqlite"), "--cpu-test"]

        def spawn(command):
            process = subprocess.Popen(command, env=env, stdout=logs, stderr=logs)
            processes.append(process)
            return process

        def request(path, body=None, auth=True, method=None):
            headers = {"Content-Type": "application/json"}
            if auth:
                headers["Authorization"] = "Bearer " + env["SOUL_TOKEN"]
            data = None if body is None else json.dumps(body).encode()
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers, method=method), timeout=10) as response:
                content = response.read()
                return json.loads(content) if content else None

        def wait_for(check, timeout=45):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                try:
                    result = check()
                    if result:
                        return result
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(.2)
            raise AssertionError("timed out waiting for service state")

        def submit(identifier, command, **shape):
            spec = json.loads((ROOT / "examples/job.json").read_text())
            spec.update(id=identifier, image="ubuntu:24.04", command=command, protected_seconds=60, **shape)
            request("/v1/jobs", spec)
            return spec

        def is_state(identifier, state):
            job = request("/v1/jobs/" + identifier)
            return job if job["state"] == state else None

        def stop(process):
            process.terminate()
            process.wait(timeout=10)

        try:
            controller = spawn(controller_cmd)
            state = wait_for(lambda: request("/v1/state"))
            controller_id = state["controller_id"]
            for identifier in ("restart", "cancel"):
                containers.add(f"soul-{controller_id}-node-1-{identifier}-1")
            try:
                request("/v1/state", auth=False)
                raise AssertionError("API accepted an unauthenticated request")
            except urllib.error.HTTPError as error:
                assert error.code == 401
            node = spawn(node_cmd)
            spec = submit(
                "restart",
                ["sh", "-ec", 'test "$SOUL_NODE_COUNT" = 1; echo soul-started; sleep 12; echo soul-finished'],
                nodes=2, fallback_nodes=[1],
            )
            wait_for(lambda: is_state("restart", "running"))
            request("/v1/jobs", spec)  # lost submission acknowledgement
            scheduling = "/v1/nodes/node-1/scheduling"
            drained = request(scheduling, {"schedulable": False}, method="PUT")
            assert drained == {"schedulable": False, "allocations": [
                {"job": "restart", "attempt": 1, "gpus": [0]}
            ]}
            submit("cancel", ["sleep", "120"])
            stop(node)
            stop(controller)
            controller = spawn(controller_cmd)
            wait_for(lambda: request("/v1/state"))
            node = spawn(node_cmd)
            job = wait_for(lambda: is_state("restart", "completed"))
            assert len(job["attempts"]) == 1
            attempt = job["attempts"][0]
            assert attempt["placement"] == [{"node": "node-1", "gpus": [0]}]
            charged = request("/v1/pools/b300/usage")["accounts"]
            assert charged["research"] == charged["training"] == attempt["ended_at"] - attempt["started_at"]
            output = subprocess.check_output(["docker", "logs", f"soul-{controller_id}-node-1-restart-1"], text=True)
            assert output.count("soul-started") == 1 and "soul-finished" in output
            assert request(scheduling) == {"schedulable": False, "allocations": []}
            assert is_state("cancel", "queued")
            request(scheduling, {"schedulable": True}, method="PUT")
            wait_for(lambda: is_state("cancel", "running"))
            request("/v1/jobs/cancel/cancel", {})
            job = wait_for(lambda: is_state("cancel", "cancelled"))
            assert job["attempts"][0]["ended_at"] is not None
            container_state = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Running}}", f"soul-{controller_id}-node-1-cancel-1"], text=True).strip()
            assert container_state == "false"
            print("PASS: fallback launch, charged GPU time, authenticated API, idempotent submit, durable drain across restart, resume and confirmed cancellation")
        except BaseException:
            logs.flush()
            logs.seek(0)
            print(logs.read())
            raise
        finally:
            for process in processes:
                if process.poll() is None:
                    stop(process)
            for container in containers:
                subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logs.close()


if __name__ == "__main__":
    main()
