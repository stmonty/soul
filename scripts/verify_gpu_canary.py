#!/usr/bin/env python3
"""Verify a Soul canary using its controller result, rank-zero log and host probes."""
import argparse
import json
from pathlib import Path
import sys
from uuid import UUID


def receipt_for(job, lines):
    receipt = None
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # Docker logs may also contain launcher diagnostics.
        if isinstance(entry, dict) and entry.get("job") == job and "nccl_sum" in entry:
            if receipt is not None:
                raise ValueError(f"duplicate application receipt for {job}")
            receipt = entry
    if receipt is None:
        raise ValueError(f"missing application receipt for {job}")
    return receipt


def verify(run, log, probes):
    job = run["job"]
    if run["state"] != "completed":
        raise ValueError(f"{job} did not complete")
    receipt = receipt_for(job, log)
    inventory = {}
    boots = {}
    for node, report in probes:
        probe = report["docker_gpu_assignment"]
        if not report["basic_prerequisites_pass"] or not probe["ok"]:
            raise ValueError(f"GPU probe failed on {node}")
        key = (node, probe["index"])
        if key in inventory:
            raise ValueError(f"duplicate probe for {key}")
        boot = UUID(probe["boot_id"])
        if boots.setdefault(node, boot) != boot:
            raise ValueError(f"probes from different boots for {node}")
        inventory[key] = UUID(probe["host_uuid"].removeprefix("GPU-"))

    placement = run["placement"]
    if len({boots[member["node"]] for member in placement}) != len(placement):
        raise ValueError("multiple assigned nodes share a host boot ID")
    world = sum(len(member["gpus"]) for member in placement)
    ranks = [UUID(value) for value in receipt["gpu_uuids_by_rank"]]
    if world < 1 or len(ranks) != world or len(set(ranks)) != world or receipt["world_size"] != world:
        raise ValueError("rank count or device uniqueness disagrees with the allocation")
    if receipt["nccl_sum"] != world * (world + 1) // 2:
        raise ValueError("incorrect NCCL sum")

    start = 0
    for member in placement:
        node = member["node"]
        end = start + len(member["gpus"])
        expected = {inventory[node, index] for index in member["gpus"]}
        if set(ranks[start:end]) != expected:
            raise ValueError(f"rank UUIDs disagree with assigned GPU indices on {node}")
        start = end
    return {"verified_job": job, "world_size": world, "image": run["image"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="JSON output of gpu_canary.py")
    parser.add_argument("rank_zero_log", type=Path)
    parser.add_argument("--probe", nargs=2, action="append", required=True, metavar=("NODE", "REPORT"),
                        help="Node ID and preflight JSON file, repeat for each assigned GPU")
    args = parser.parse_args()
    probes = ((node, json.loads(Path(path).read_text())) for node, path in args.probe)
    with args.rank_zero_log.open() as log:
        print(json.dumps(verify(json.loads(args.run.read_text()), log, probes)))


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, OSError) as error:
        sys.exit(f"Canary verification failed: {error}")
