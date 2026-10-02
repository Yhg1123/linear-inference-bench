import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from benchmark import select_quantization_engine
from diagnose_qa_quantization import compare_linear, rounded_mlp_weights


class QuantizationDiagnosisTests(unittest.TestCase):
    def setUp(self):
        old = torch.backends.quantized.engine
        self.addCleanup(setattr, torch.backends.quantized, "engine", old)
        torch.backends.quantized.engine = select_quantization_engine(torch.backends.quantized.supported_engines)
        torch.manual_seed(12)

    def test_replay_preserves_original_and_tokenwise_singleton_agrees(self):
        layer = nn.Linear(8, 5).eval()
        x = torch.randn(1, 3, 8)
        before = layer.weight.detach().clone()
        rows = compare_linear(layer, x)
        self.assertTrue(torch.equal(before, layer.weight))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["finite"] for r in rows))
        for row in rows:
            if row["scope"] == "last_token":
                self.assertAlmostEqual(row["dynamic_relative_l2"], row["tokenwise_dynamic_relative_l2"], places=6)

    def test_weight_control_restores_parameters_even_after_failure(self):
        model = nn.Module()
        model.block = nn.Module()
        model.block.mlp = nn.Sequential(nn.Linear(8, 4))
        model.output = nn.Linear(4, 2)
        weight = model.block.mlp[0].weight
        output = model.output.weight
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with rounded_mlp_weights(model, True) as count:
                self.assertEqual(count, 1)
                self.assertIsNot(model.block.mlp[0].weight, weight)
                self.assertIs(model.output.weight, output)
                self.assertIsInstance(model.block.mlp[0], nn.Linear)
                raise RuntimeError("deliberate")
        self.assertIs(model.block.mlp[0].weight, weight)
