import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from benchmark import TinyMLP
from profile_workloads import request


class RequestTests(unittest.TestCase):
    def test_cpu_request_preserves_reference(self):
        model = TinyMLP().eval()
        x = torch.randn(3, 768)
        with torch.inference_mode():
            actual = request(model, x, "cpu_fp32")
            torch.testing.assert_close(actual, model(x))
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_request_returns_cpu_fp32(self):
        model = TinyMLP().eval().cuda().half()
        x = torch.randn(3, 768)
        with torch.inference_mode():
            actual = request(model, x, "cuda_fp16")
            expected = model(x.cuda().half()).float().cpu()
            torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float32)
