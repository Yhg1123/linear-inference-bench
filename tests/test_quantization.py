import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

from benchmark import TinyMLP, accuracy


class QuantizationTests(unittest.TestCase):
    def test_dynamic_quantization_changes_linear_layers_and_keeps_output_close(self):
        torch.manual_seed(9)
        model = TinyMLP().eval()
        quantized = quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8).eval()
        self.assertIsInstance(quantized.net[0], torch.ao.nn.quantized.dynamic.Linear)
        x = torch.randn(4, 768)
        with torch.inference_mode():
            metrics = accuracy(model(x), quantized(x))
        self.assertLess(metrics["relative_l2_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
