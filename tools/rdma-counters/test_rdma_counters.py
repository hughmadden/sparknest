"""The reference decoder reads the frames the daemon's encoder writes
(testdata/frames.bin, checked byte for byte by nest-api's tests)."""
import importlib.machinery
import importlib.util
import io
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    loader = importlib.machinery.SourceFileLoader("rdma_counters", str(HERE / "sparknest-rdma-counters"))
    spec = importlib.util.spec_from_loader("rdma_counters", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class FixtureTest(unittest.TestCase):
    def test_decodes_the_shared_fixture(self):
        m = load()
        frames = list(m.frames(io.BytesIO((HERE / "testdata" / "frames.bin").read_bytes())))
        self.assertEqual([f["type"] for f in frames], ["hello", "counters"])
        h, c = frames
        self.assertEqual(h["schema"], 1)
        self.assertEqual(h["start_id"], 0x0123456789ABCDEF)
        self.assertEqual(h["start_epoch_ns"], 1_791_000_000_000_000_000)
        self.assertEqual(h["boot_id"], "00010203-0405-0607-0809-0a0b0c0d0e0f")
        self.assertEqual(h["generation"], 1)
        self.assertEqual(h["ports"], [
            {"device": "mlx5_0", "port": 1, "netdev": "enp1s0np0",
             "addrs": ["10.55.0.22", "10.55.1.22"], "path_mtu": 4096},
            {"device": "rocep1s0f0", "port": 1, "netdev": "enp1s0f0np0",
             "addrs": ["10.55.0.1"], "path_mtu": 1024},
        ])
        self.assertEqual((c["seq"], c["mono_ns"], c["wall_ns"], c["generation"]),
                         (42, 123_456_789, 1_791_000_000_500_000_000, 1))
        self.assertEqual(c["values"][0], {
            "bytes_sent": 7 << 40, "bytes_received": 7, "ops_sent": 2,
            "ops_received": 3, "packets_sent": 4, "packets_received": 5})
        self.assertEqual(c["values"][1]["bytes_sent"], 9 << 40)


if __name__ == "__main__":
    unittest.main()
