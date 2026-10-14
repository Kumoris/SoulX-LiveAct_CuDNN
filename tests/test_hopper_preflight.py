"""A migration must not accidentally continue on the retired GPU route."""
import unittest

from scripts.hopper_preflight import validate_device, validate_pair


class HopperPreflightTests(unittest.TestCase):
    def device(self, rank=0, name="NVIDIA H100 80GB HBM3", host="hopper-1"):
        return dict(rank=rank, name=name, hostname=host,
                    capability=[9, 0], memory_bytes=80 * 2**30)

    def test_matching_h100_or_h200_pair(self):
        for name in ["NVIDIA H100 80GB HBM3", "NVIDIA H200"]:
            validate_pair([self.device(0, name), self.device(1, name)])

    def test_a800_and_incorrect_capability_are_rejected(self):
        for name, capability in [("NVIDIA A800-SXM4-80GB", (8, 0)),
                                 ("NVIDIA H100", (8, 0))]:
            with self.assertRaises(ValueError):
                validate_device(name, capability, 80 * 2**30)

    def test_insufficient_memory_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_device("NVIDIA H100", (9, 0), 40 * 2**30)

    def test_wrong_rank_count_or_duplicate_rank_is_rejected(self):
        for devices in [[self.device()], [self.device(), self.device()]]:
            with self.assertRaises(ValueError):
                validate_pair(devices)

    def test_cross_node_or_mixed_pair_is_rejected(self):
        for second in [self.device(1, host="hopper-2"), self.device(1, name="NVIDIA H200")]:
            with self.assertRaises(ValueError):
                validate_pair([self.device(), second])


if __name__ == "__main__":
    unittest.main()
