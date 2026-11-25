"""Exercise an acknowledged checkpoint and one-to-two-node restore on local Docker."""
import json
import os
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("SOUL_BIN_DIR", ROOT / "target/debug")).resolve()


def main():
    with tempfile.TemporaryDirectory(prefix="soul-resize-") as workdir:
        directory = Path(workdir)
        checkpoint = directory / "checkpoint.json"

        class Artifact(BaseHTTPRequestHandler):
            def do_PUT(self):
                if self.path != "/checkpoint":
                    return self.send_error(404)
                with checkpoint.open("wb") as output:
                    output.write(self.rfile.read(int(self.headers["Content-Length"])))
                    output.flush()
                    os.fsync(output.fileno())
                self.send_response(201)
                self.end_headers()

            def do_GET(self):
                if self.path != "/checkpoint" or not checkpoint.exists():
                    return self.send_error(404)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(checkpoint.read_bytes())

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Artifact) as artifacts, (directory / "services.log").open("w+") as logs:
            threading.Thread(target=artifacts.serve_forever, daemon=True).start()
            artifact = f"http://127.0.0.1:{artifacts.server_port}/checkpoint"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                address = f"127.0.0.1:{sock.getsockname()[1]}"
            url = "http://" + address
            token = secrets.token_hex(32)
            env = dict(os.environ, SOUL_TOKEN=token)
            processes = []
            names = []

            def start(args):
                proc = subprocess.Popen(args, env=env, stdout=logs, stderr=logs)
                processes.append(proc)
                return proc

            def stop(proc):
                proc.terminate()
                proc.wait(timeout=10)

            def controller():
                return start([BIN / "soul", "serve", directory / "controller.sqlite",
                              ROOT / "examples/two-node.json", address])

            def node(n):
                return start([BIN / "soul-node", url, f"node-{n}",
                              directory / f"node-{n}.sqlite", "--cpu-test", "--host-network"])

            def api(path, payload=None, credential=token):
                data = None if payload is None else json.dumps(payload).encode()
                req = Request(url + path, data=data, headers={"Authorization": "Bearer " + credential,
                                                            "Content-Type": "application/json"})
                with urlopen(req, timeout=5) as response:
                    body = response.read()
                    return json.loads(body) if body else None

            def wait(predicate):
                end = time.monotonic() + 45
                while time.monotonic() < end:
                    try:
                        result = predicate()
                        if result:
                            return result
                    except HTTPError:
                        raise
                    except (URLError, ConnectionError):
                        pass
                    time.sleep(.15)
                raise RuntimeError("timed out waiting for resize smoke condition")

            def job():
                return api("/v1/jobs/trainer")

            try:
                server = controller()
                state = wait(lambda: api("/v1/state"))
                prefix = f"soul-{state['controller_id']}"
                for n, a in [(1, 1), (1, 2), (2, 2)]:
                    names.append(f"{prefix}-node-{n}-trainer-{a}")
                first = node(1)
                api("/v1/jobs", {"id": "trainer", "pool": "test", "account": "research",
                    "image": "python:3.12-slim", "command": ["python", "-c",
                        (ROOT / "examples/checkpoint_trainer.py").read_text(), artifact],
                    "nodes": 2, "fallback_nodes": [1], "gpus_per_node": 1,
                    "protected_seconds": 0, "resumable": False, "resize_policy": "cost_aware"})
                wait(lambda: job()["state"] == "running")
                node(2)
                wait(lambda: "node-2" in api("/v1/state")["checkpoint_nodes"])
                container_env = json.loads(subprocess.check_output(
                    ["docker", "inspect", "--format", "{{json .Config.Env}}", names[0]], text=True))
                scoped_token = next(entry.split("=", 1)[1] for entry in container_env
                                    if entry.startswith("SOUL_RUNTIME_TOKEN="))
                assert scoped_token != token
                assert api("/v1/runtime/trainer/1", credential=scoped_token) is None
                # Exercise the application credential for both a hold and a growth proposal.
                profile_url = "/v1/runtime/trainer/1/profile"
                profile = {"sequence": 1, "valid_for_seconds": 120, "remaining_work": 1000,
                           "rates": {"1": 1, "2": 2}, "checkpoint_seconds": 2, "restore_seconds": 600}
                api(profile_url, profile, scoped_token)
                held = api("/v1/jobs/trainer/explain")["growth_forecast"]
                assert held["status"] == "compared" and held["recommended_nodes"] is None
                stop(first)
                profile.update(sequence=2, restore_seconds=1)
                api(profile_url, profile, scoped_token)
                api(profile_url, profile, scoped_token)  # retry after a lost response
                request = {"id": "auto-grow", "attempt": 1, "nodes": 2}
                wait(lambda: job()["state"] == "stopping")
                assert job()["attempts"][0]["ended_at"] is None
                stop(server)
                controller()
                wait(lambda: job()["state"] == "stopping")
                assert api("/v1/jobs/trainer/resize", request)["outcome"]["status"] == "ready"
                node(1)
                wait(lambda: job()["state"] == "completed")
                a, b = job()["attempts"]
                assert len(a["placement"]) == 1 and len(b["placement"]) == 2
                assert a["ended_at"] <= b["started_at"]
                evidence = a["resizes"][0]["estimate"]
                assert evidence["sequence"] == 2 and evidence["stay_seconds"] > evidence["resize_seconds"]
                saved = json.loads(checkpoint.read_text())
                assert saved["step"] > 0 and saved["step"] == a["resizes"][0]["outcome"]["checkpoint"]["progress"]
                for name in names[1:]:
                    output = subprocess.check_output(["docker", "logs", name], text=True)
                    assert f"restored step={saved['step']} total={saved['total']} nodes=2" in output
                for path in ["/v1/state", "/v1/runtime/trainer/1", "/v1/runtime/trainer/2"]:
                    try:
                        api(path, credential=scoped_token)
                    except HTTPError as e:
                        assert e.code == 401
                    else:
                        raise AssertionError("API accepted a stale or out-of-scope attempt credential")
                try:
                    api("/v1/runtime/trainer/2/profile", profile, scoped_token)
                except HTTPError as error:
                    assert error.code == 401
                else:
                    raise AssertionError("replacement accepted old runtime profile credential")
                print("PASS: cost-aware hold and growth, durable checkpoint, controller restart with old GPU held, both replacement ranks restored progress")
            except BaseException:
                logs.flush()
                logs.seek(0)
                print(logs.read())
                raise
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        stop(proc)
                for name in names:
                    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                artifacts.shutdown()


if __name__ == "__main__":
    main()
