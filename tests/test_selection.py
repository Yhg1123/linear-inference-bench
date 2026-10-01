import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_profile import choose
from bench_utils import summarize, percentile


class SelectionTests(unittest.TestCase):
    def row(self, name, **changes):
        return dict(workload="a", option=name, status="ok", successful_trials=3, expected_trials=3,
                    median_ms=1., p95_ms=2., relative_l2_error=0.001, spread_ratio=1.2,
                    peak_extra_mib=1., model_state_mib=4., **changes)

    def select(self, rows, **limits):
        return choose(rows, ["workload"], "option", max_error=0.01, **limits)

    def test_fast_but_bad_error_is_excluded(self):
        fast = self.row("fast")
        fast.update(median_ms=0.1, relative_l2_error=0.1)
        self.assertEqual(self.select([fast, self.row("safe")])[0]["selected"], "safe")

    def test_no_match_retains_workload_and_reasons(self):
        decision = self.select([self.row("only")], max_memory=0.1)[0]
        self.assertEqual(decision["status"], "no_match")
        self.assertEqual(decision["workload"], {"workload": "a"})
        self.assertIn("peak_extra_mib_constraint", decision["rejected"][0]["reasons"])

    def test_nan_or_missing_trial_cannot_win(self):
        bad = self.row("nan")
        bad["relative_l2_error"] = "nan"
        partial = self.row("partial")
        partial["successful_trials"] = 2
        self.assertEqual(self.select([bad, partial])[0]["status"], "no_match")

    def test_latency_tail_and_variance_can_exclude_an_option(self):
        self.assertEqual(self.select([self.row("a")], max_p95=1.5)[0]["status"], "no_match")
        self.assertEqual(self.select([self.row("a")], max_spread=1.1)[0]["status"], "no_match")

    def test_workloads_do_not_leak_choices(self):
        a, b = self.row("a"), self.row("b")
        b["workload"] = "b"
        b["status"] = "unsupported"
        self.assertEqual([x["status"] for x in self.select([a, b])], ["selected", "no_match"])

    def test_summary_does_not_hide_an_unsupported_trial(self):
        success = dict(workload="a", option="x", trial=0, status="ok", median_ms=1., p95_ms=2.,
                       relative_l2_error=0., max_abs_error=0., items_per_request=1)
        failure = dict(workload="a", option="x", trial=1, status="unsupported", reason="no kernel")
        result = summarize([success, failure], ["workload"], "option", 2)[0]
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["successful_trials"], 1)

    def test_p95_uses_nearest_rank(self):
        self.assertEqual(percentile(list(range(1, 21)), 0.95), 19)

    def test_task_accuracy_constraint_requires_measured_quality(self):
        candidate = self.row("quantized")
        self.assertEqual(self.select([candidate], max_accuracy_drop=1.)[0]["status"], "no_match")
        candidate["accuracy_drop_pp"] = 2.
        self.assertEqual(self.select([candidate], max_accuracy_drop=1.)[0]["status"], "no_match")
        candidate["accuracy_drop_pp"] = 0.5
        self.assertEqual(self.select([candidate], max_accuracy_drop=1.)[0]["status"], "selected")

    def test_disallowed_device_variant_cannot_win(self):
        gpu, cpu = self.row("cuda_fp16"), self.row("cpu_fp32")
        gpu["median_ms"] = 0.01
        result = self.select([gpu, cpu], only_options=["cpu_fp32"])[0]
        self.assertEqual(result["selected"], "cpu_fp32")
