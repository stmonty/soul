"""One torchrun worker per assigned GPU. Rank zero emits the collective receipt."""
from datetime import timedelta
import json
import os
import sys
from uuid import UUID

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    devices = int(os.environ["SOUL_GPUS_PER_NODE"])
    nodes = int(os.environ["SOUL_NODE_COUNT"])
    node_rank = int(os.environ["SOUL_NODE_RANK"])
    if torch.cuda.device_count() != devices:
        raise RuntimeError(f"expected {devices} visible GPUs, found {torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != nodes * devices or rank != node_rank * devices + local_rank:
            raise RuntimeError(f"unexpected rank {rank} or world size {world}")

        values = torch.full((65536,), rank + 1, dtype=torch.int64, device="cuda")
        dist.all_reduce(values)
        expected = world * (world + 1) // 2
        if not (values == expected).all().item():
            raise RuntimeError("incorrect NCCL sum")

        # Record the physical identities in global rank order for inventory comparison.
        gpu = UUID(str(torch.cuda.get_device_properties(local_rank).uuid))
        identity = torch.tensor(list(gpu.bytes), dtype=torch.uint8, device="cuda")
        identities = torch.empty(world * 16, dtype=torch.uint8, device="cuda")
        dist.all_gather_into_tensor(identities, identity)
        uuids = [str(UUID(bytes=bytes(row))) for row in identities.view(world, 16).tolist()]
        if len(set(uuids)) != world:
            raise RuntimeError("multiple ranks used the same GPU")
        if rank == 0:
            print(json.dumps({"job": sys.argv[1], "world_size": world, "nccl_sum": expected,
                              "gpu_uuids_by_rank": uuids}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
