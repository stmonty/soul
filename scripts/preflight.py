#!/usr/bin/env python3
"""Inspect host prerequisites. Optionally check a Soul GPU assignment inside Docker."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def command(args):
    executable = shutil.which(args[0])
    if not executable:
        return {"ok": False, "error": f"{args[0]} not installed"}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"ok": result.returncode == 0,
                "output": result.stdout.strip()[:8192],
                "error": result.stderr.strip()[:8192]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}


def inspect(path):
    disk = shutil.disk_usage(path)
    memory = {}
    try:
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    except (OSError, ValueError):
        pass
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_total": memory.get("MemTotal", "unknown").strip(),
        "disk": {"path": str(Path(path).resolve()), "free_bytes": disk.free},
        "docker": command(["docker", "info", "--format", "{{json .ServerVersion}}"]),
        "gpus": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader"]),
        "gpu_topology": command(["nvidia-smi", "topo", "-m"]),
        "kvm": {"exists": Path("/dev/kvm").exists(), "read_write": os.access("/dev/kvm", os.R_OK | os.W_OK)},
        "firecracker": command(["firecracker", "--version"]),
        "rdma_devices": sorted(p.name for p in Path("/sys/class/infiniband").glob("*")),
        "rust": command(["rustc", "--version"]),
    }


def probe_gpu(index, image):
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError as error:
        return {"ok": False, "error": str(error)}
    query = ["--query-gpu=uuid", "--format=csv,noheader"]
    host = command(["nvidia-smi", f"--id={index}", *query])
    if not host["ok"]:
        return {"ok": False, "host": host}
    container = command(["docker", "run", "--rm", "--pull=never", "--network=none",
                         "--gpus", f'"device={index}"', "--entrypoint", "nvidia-smi", image, *query])
    expected = host["output"]
    return {"ok": container["ok"] and expected.startswith("GPU-")
            and len(expected.splitlines()) == 1 and container["output"] == expected,
            "boot_id": boot_id, "index": index, "host_uuid": expected, "container": container}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", choices=["cpu", "container", "gpu", "microvm"], default="cpu")
    parser.add_argument("--disk", default=".", help="Directory containing planned persistent service state")
    parser.add_argument("--probe-docker-gpu", type=int, metavar="INDEX",
                        help="Start a temporary container and verify its sole GPU UUID against the host")
    parser.add_argument("--probe-image", default="ubuntu:24.04", help="Locally cached image for the GPU probe")
    args = parser.parse_args()
    if args.probe_docker_gpu is not None and args.probe_docker_gpu < 0:
        parser.error("GPU index must be nonnegative")
    report = inspect(args.disk)
    required = {
        "cpu": True,
        "container": report["docker"]["ok"],
        "gpu": report["docker"]["ok"] and report["gpus"]["ok"],
        "microvm": report["kvm"]["read_write"] and report["firecracker"]["ok"],
    }
    report["required_profile"] = args.require
    passed = required[args.require]
    if args.probe_docker_gpu is not None:
        probe = probe_gpu(args.probe_docker_gpu, args.probe_image)
        report["docker_gpu_assignment"] = probe
        passed = passed and probe["ok"]
    report["basic_prerequisites_pass"] = passed
    report["limits"] = "CUDA computation, distributed execution and KVM ioctls require separate workload tests."
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
