import json
import unittest
from uuid import UUID

from scripts.verify_gpu_canary import verify


class CanaryEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.run = {"job": "canary-example", "state": "completed", "image": "test-image",
                    "placement": [{"node": "a", "gpus": [2, 0]}, {"node": "b", "gpus": [0, 1]}]}
        self.uuids = [str(UUID(int=i)) for i in range(1, 5)]
        self.probes = [
            (node, {"basic_prerequisites_pass": True, "docker_gpu_assignment": {
                "ok": True, "boot_id": str(UUID(int=ord(node))), "index": index,
                "host_uuid": "GPU-" + identity}})
            for (node, index), identity in zip([("a", 0), ("a", 2), ("b", 0), ("b", 1)], self.uuids)
        ]

    def log(self, ranks):
        return [json.dumps({"job": self.run["job"], "world_size": 4,
                            "nccl_sum": 10, "gpu_uuids_by_rank": ranks})]

    def test_accepts_cuda_reordering_within_a_node(self):
        result = verify(self.run, ["torchrun diagnostics", *self.log(self.uuids)], self.probes)
        self.assertEqual(result["world_size"], 4)

    def test_rejects_right_gpus_on_wrong_nodes(self):
        with self.assertRaisesRegex(ValueError, "indices on a"):
            verify(self.run, self.log(self.uuids[::-1]), self.probes)

    def test_rejects_two_daemons_on_one_host_boot(self):
        for _, report in self.probes:
            report["docker_gpu_assignment"]["boot_id"] = self.uuids[0]
        with self.assertRaisesRegex(ValueError, "share a host boot"):
            verify(self.run, self.log(self.uuids), self.probes)

    def test_requires_distinct_gpus_and_application_receipt(self):
        with self.assertRaisesRegex(ValueError, "uniqueness"):
            verify(self.run, self.log([self.uuids[0]] * 4), self.probes)
        with self.assertRaisesRegex(ValueError, "missing application receipt"):
            verify(self.run, ["entrypoint exited zero"], self.probes)
