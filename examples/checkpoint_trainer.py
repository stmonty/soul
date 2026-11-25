"""Tiny cooperating workload. Its argument is a shared HTTP GET/PUT artifact URL."""
import json
import os
import signal
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
reference = os.getenv("SOUL_CHECKPOINT_REF")
if reference:
    with urlopen(reference, timeout=5) as response:
        saved = json.load(response)
    step = int(os.environ["SOUL_CHECKPOINT_PROGRESS"])
    assert step == saved["step"] and saved["total"] == sum(range(1, step + 1))
    print(f"restored step={step} total={saved['total']} nodes={os.environ['SOUL_NODE_COUNT']}", flush=True)
    sys.exit(0)

# This example starts on a single node. In a distributed trainer the coordinator
# must establish that every shard is durable before acknowledging the checkpoint.
url = os.environ["SOUL_RUNTIME_URL"]
headers = {"Authorization": "Bearer " + os.environ["SOUL_RUNTIME_TOKEN"], "Content-Type": "application/json"}
def runtime_request(body=None):
    data = None if body is None else json.dumps(body).encode()
    while True:
        try:
            with urlopen(Request(url, data=data, headers=headers), timeout=5) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 409:  # Expired or superseded while the checkpoint was being written.
                return None
            if error.code < 500:
                raise
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.25)  # Retry the identical acknowledgement after a lost response.


step = total = 0
while True:
    intent = runtime_request()
    if intent:
        artifact = sys.argv[1]
        data = json.dumps({"step": step, "total": total}).encode()
        with urlopen(Request(artifact, method="PUT", data=data), timeout=5):
            pass
        reply = {"id": intent["id"], "status": "ready", "checkpoint": {"reference": artifact, "progress": step}}
        if runtime_request(reply):
            while True:
                signal.pause()
    step += 1
    total += step
    time.sleep(0.1)
