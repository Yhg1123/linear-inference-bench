import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

from benchmark import TinyMLP, accuracy, select_quantization_engine


class QuantizationTests(unittest.TestCase):
    def test_onednn_only_build_can_select_dynamic_quantization(self):
        self.assertEqual(select_quantization_engine(["onednn"]), "onednn")

    def test_original_engine_preference_is_preserved(self):
        self.assertEqual(select_quantization_engine(["onednn", "qnnpack", "fbgemm", "x86"]), "x86")
        self.assertEqual(select_quantization_engine(["onednn", "fbgemm"]), "fbgemm")
        self.assertEqual(select_quantization_engine(["onednn", "qnnpack"]), "qnnpack")

    def test_missing_dynamic_quantization_engine_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "No supported dynamic quantization backend"):
            select_quantization_engine(["none"])

    def test_dynamic_quantization_changes_linear_layers_and_keeps_output_close(self):
        torch.manual_seed(9)
        previous_engine = torch.backends.quantized.engine
        self.addCleanup(setattr, torch.backends.quantized, "engine", previous_engine)
        torch.backends.quantized.engine = select_quantization_engine(torch.backends.quantized.supported_engines)
        model = TinyMLP().eval()
        quantized = quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8).eval()
        self.assertIsInstance(quantized.net[0], torch.ao.nn.quantized.dynamic.Linear)
        x = torch.randn(4, 768)
        with torch.inference_mode():
            metrics = accuracy(model(x), quantized(x))
        self.assertLess(metrics["relative_l2_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
