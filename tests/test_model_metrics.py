import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluate_model import classification_metrics


class ModelMetricTests(unittest.TestCase):
    def test_same_accuracy_can_hide_changed_predictions(self):
        labels = torch.tensor([0, 1, 0, 1])
        baseline = torch.tensor([[3., 1.], [0., 2.], [0., 2.], [0., 2.]])
        candidate = torch.tensor([[1., 3.], [0., 2.], [2., 0.], [0., 2.]])
        result = classification_metrics(labels, baseline, candidate)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["changed_predictions"], 2)
        self.assertEqual(result["regressions"], 1)
        self.assertEqual(result["recoveries"], 1)
        self.assertEqual(result["accuracy_drop_pp"], 0.)
