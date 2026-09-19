import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from llmeval import bench, cli, compare, report, stats
from llmeval.fakeserver import FakeServer

ROOT = os.path.dirname(os.path.dirname(__file__))
FIX = os.path.join(ROOT, "tests", "fixtures")


class StatsTests(unittest.TestCase):
    def test_percentile_and_summary(self):
        v = list(range(1, 101))
        self.assertEqual(stats.percentile(v, 50), 50)
        self.assertEqual(stats.percentile(v, 99), 99)
        self.assertEqual(stats.percentile([], 50), 0.0)
        s = stats.summarize([3, 1, 2])
        self.assertEqual((s["p50"], s["max"], s["count"]), (2, 3, 3))

    def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_data(self):
        few = [10, 12, 11, 13, 9]
        many = few * 8
        lo, hi = stats.bootstrap_ci(few)
        self.assertLessEqual(lo, 11)
        self.assertGreaterEqual(hi, 11)
        lo2, hi2 = stats.bootstrap_ci(many)
        self.assertLess(hi2 - lo2, hi - lo)
        self.assertIsNone(stats.bootstrap_ci([1, 2]))
        self.assertEqual(stats.bootstrap_ci(few), stats.bootstrap_ci(few))  # deterministic seed

    def test_interval_overlap(self):
        self.assertTrue(stats.intervals_overlap((1, 3), (2, 4)))
        self.assertFalse(stats.intervals_overlap((1, 2), (3, 4)))
        self.assertIsNone(stats.intervals_overlap(None, (1, 2)))


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = FakeServer(("127.0.0.1", 0), ttft_s=0.03, tpot_s=0.004, capacity=8,
                             info={"version": "fake-9", "tp_size": 2, "unrelated": 1}).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_single_request_measures_ttft_tpot_and_tokens(self):
        s = bench.one_request(self.srv.url, "m", "sys", "hello", 10)
        self.assertTrue(s.ok)
        self.assertEqual(s.out_tokens, 10)
        self.assertGreaterEqual(s.ttft_s, 0.03)
        self.assertLess(s.ttft_s, 0.5)
        self.assertGreaterEqual(s.tpot_s, 0.004)
        self.assertGreaterEqual(s.latency_s, s.ttft_s)

    def test_token_count_falls_back_to_chunks_without_usage(self):
        s = bench.one_request(self.srv.url, "m", "sys", "hello", 7, include_usage=False)
        self.assertEqual(s.out_tokens, 7)

    def test_matrix_run_has_repetitions_intervals_and_environment(self):
        cfg = {"endpoint": self.srv.url, "model": "m", "label": "unit", "metadata": {"gpu_model": "fake"},
               "matrix": {"concurrency": [1, 4], "input_words": [16], "max_tokens": [8]},
               "repetitions": 3, "requests_per_slot": 2, "warmup_requests": 2, "timeout_s": 10}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
        try:
            res = bench.run_matrix(bench.load_config(f.name))
        finally:
            os.unlink(f.name)
        self.assertEqual(res["schema"], bench.SCHEMA)
        self.assertEqual(res["server_info"], {"version": "fake-9", "tp_size": 2})  # only whitelisted keys kept
        self.assertEqual(len(res["cells"]), 2)
        for c in res["cells"]:
            self.assertEqual(len(c["repetitions"]), 3)
            self.assertEqual(c["summary"]["success_rate"]["mean"], 1.0)
            self.assertIsNotNone(c["summary"]["latency_p95_ms"]["ci95"])
            self.assertGreater(c["summary"]["output_tokens_per_s"]["mean"], 0)
        c1, c4 = res["cells"]
        self.assertGreater(c4["summary"]["output_tokens_per_s"]["mean"], c1["summary"]["output_tokens_per_s"]["mean"])

    def test_failures_are_counted_not_hidden(self):
        srv = FakeServer(("127.0.0.1", 0), ttft_s=0.001, tpot_s=0.001, fail_every=2).start()
        try:
            samples = [bench.one_request(srv.url, "m", "s", "u", 3) for _ in range(6)]
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(sum(s.ok for s in samples), 3)
        self.assertTrue(all(s.status == 503 for s in samples if not s.ok))
        agg = bench.aggregate(samples, 1.0)
        self.assertAlmostEqual(agg["success_rate"], 0.5)

    def test_incomplete_or_error_streams_are_failures_even_with_tokens(self):
        for kw, why in (({"omit_done": True}, "without [DONE]"), ({"error_event": True}, "error event")):
            srv = FakeServer(("127.0.0.1", 0), ttft_s=0.001, tpot_s=0.001, **kw).start()
            try:
                s = bench.one_request(srv.url, "m", "s", "u", 6)
            finally:
                srv.shutdown()
                srv.server_close()
            self.assertFalse(s.ok, why)
            self.assertGreater(s.out_tokens, 0)      # tokens arrived, but the request is still not a success
            self.assertIn(why.split()[-1], s.error)

    def test_unreachable_endpoint_is_a_failed_sample(self):
        s = bench.one_request("http://127.0.0.1:9", "m", "s", "u", 3, timeout=1)
        self.assertFalse(s.ok)
        self.assertTrue(s.error)

    def test_prompts_share_prefix_within_group_and_differ_in_user_text(self):
        s0, u0 = bench.build_prompts(0, 4, 20, 1)
        s4, u4 = bench.build_prompts(4, 4, 20, 1)
        s1, _ = bench.build_prompts(1, 4, 20, 1)
        self.assertEqual(s0, s4)
        self.assertNotEqual(s0, s1)
        self.assertNotEqual(u0, u4)
        self.assertEqual(len(u0.split()), 20)

    def test_config_validation(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"endpoint": "http://x", "model": "m", "matrix": {"concurrency": []}}, f)
        try:
            with self.assertRaises(ValueError):
                bench.load_config(f.name)
        finally:
            os.unlink(f.name)


def result(label, meta, p50, p95, ci50=None, tps=None, conc=8):
    s = {"success_rate": {"mean": 1.0, "ci95": None},
         "latency_p50_ms": {"mean": p50, "ci95": ci50}, "latency_p95_ms": {"mean": p95, "ci95": None}}
    if tps:
        s["output_tokens_per_s"] = {"mean": tps, "ci95": None}
    return {"label": label, "metadata": meta, "cells": [{"concurrency": conc, "input_words": None, "max_tokens": None, "summary": s}]}


class CompareTests(unittest.TestCase):
    def test_real_4090_vs_a100_tp_comparison_is_flagged_as_confounded(self):
        runs = {r["label"]: r for r in compare.import_runs_csv(os.path.join(ROOT, "data", "real_gpu_runs.csv"))}
        c = compare.compare(runs["4090_tp4"], runs["a100_tp4"])
        self.assertTrue(c["confounded"])
        names = {k for k, _, _ in c["factors"]}
        self.assertTrue({"nccl_p2p_disabled", "engine_version", "driver", "deployment", "interconnect", "gpu_model"} <= names)
        self.assertIn("CONFOUNDED", c["verdict"])
        p50 = next(r for r in c["rows"] if r["metric"] == "latency_p50_ms")
        self.assertAlmostEqual(p50["improvement"], 4836 / 2754, places=3)   # the ratio is reported, but flagged
        self.assertIn("CONFOUNDED", compare.render(c))

    def test_dp_comparison_has_no_p2p_factor_but_is_still_confounded(self):
        runs = {r["label"]: r for r in compare.import_runs_csv(os.path.join(ROOT, "data", "real_gpu_runs.csv"))}
        c = compare.compare(runs["4090_dp4"], runs["a100_dp4"])
        names = {k for k, _, _ in c["factors"]}
        self.assertTrue(c["confounded"])
        self.assertIn("engine_version", names)

    def test_single_factor_is_attributable(self):
        base = {"gpu_model": "X", "engine_version": "1", "nccl_p2p_disabled": "yes"}
        a = result("a", base, 100, 200)
        b = result("b", {**base, "nccl_p2p_disabled": "no"}, 60, 120)
        c = compare.compare(a, b)
        self.assertFalse(c["confounded"])
        self.assertEqual([k for k, _, _ in c["factors"]], ["nccl_p2p_disabled"])
        self.assertIn("attributed", c["verdict"])
        self.assertAlmostEqual(next(r for r in c["rows"] if r["metric"] == "latency_p50_ms")["improvement"], 100 / 60)

    def test_identical_configuration_and_noise(self):
        a = result("a", {"gpu_model": "X"}, 100, 200, ci50=(90, 110))
        b = result("b", {"gpu_model": "X"}, 104, 205, ci50=(95, 115))
        c = compare.compare(a, b)
        self.assertEqual(c["factors"], [])
        self.assertTrue(next(r for r in c["rows"] if r["metric"] == "latency_p50_ms")["noise"])
        self.assertIn("noise", c["verdict"])

    def test_higher_is_better_metrics_invert_the_ratio(self):
        a = result("a", {"gpu_model": "X"}, 1, 1, tps=100)
        b = result("b", {"gpu_model": "X"}, 1, 1, tps=250)
        row = next(r for r in compare.compare(a, b, ["output_tokens_per_s"])["rows"])
        self.assertAlmostEqual(row["improvement"], 2.5)


class ReportTests(unittest.TestCase):
    def test_cost_formula(self):
        # 4 GPUs at $2/h sustaining 1000 tokens/s: 8 $/h / 3.6e6 tokens/h * 1e6 = 2.2222 $ per 1M tokens
        self.assertAlmostEqual(report.cost_per_million_tokens(4, 2.0, 1000), 8 / 3.6, places=4)
        self.assertIsNone(report.cost_per_million_tokens(4, 2.0, 0))

    def test_report_marks_slo_and_ranks_cost(self):
        def r(label, p95, tps, gpus, price):
            x = result(label, {"gpu_count": gpus, "gpu_hourly_usd": price}, p95 / 2, p95, tps=tps)
            x["cells"][0]["summary"]["latency_p99_ms"] = {"mean": p95 * 1.2, "ci95": None}
            return x
        md = report.render_markdown([r("cheap-but-slow", 900, 2000, 4, 1.0), r("fast", 400, 1500, 4, 3.0)], slo_p95_ms=500)
        self.assertIn("FAIL", md)
        self.assertIn("pass", md)
        self.assertIn("Cheapest cell that meets the SLO", md)
        ranked = md.split("Cheapest cell that meets the SLO")[1]
        self.assertIn("| 1 | fast |", ranked)          # the only cell that meets the SLO
        self.assertNotIn("cheap-but-slow", ranked)

    def test_plot_writes_a_file(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        r = result("a", {}, 1, 1)
        r["cells"][0]["summary"]["latency_p95_ms"] = {"mean": 5.0, "ci95": (4.0, 6.0)}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.png")
            report.plot_latency([r], path)
            self.assertGreater(os.path.getsize(path), 1000)


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(list(argv))
        return code, buf.getvalue()

    def test_size_topo_doctor(self):
        code, out = self.run_cli("size", "--model", "qwen2.5-32b", "--concurrency", "20", "--context", "512")
        self.assertEqual(code, 0)
        self.assertIn("RTX 4090", out)
        code, out = self.run_cli("topo", os.path.join(FIX, "topo_rtx4090_pcie.txt"))
        self.assertIn("interconnect: pcie", out)
        code, out = self.run_cli("doctor", os.path.join(FIX, "log_rtx4090_startup.txt"))
        self.assertEqual(code, 1)
        self.assertIn("NCCL", out)
        code, out = self.run_cli("doctor", os.path.join(FIX, "log_healthy.txt"))
        self.assertEqual(code, 0)
        code, out = self.run_cli("doctor", "--nvidia-smi", os.path.join(FIX, "nvidia_smi_banner.txt"), "--image-cuda", "13.0")
        self.assertEqual(code, 1)
        self.assertIn("image needs CUDA 13.0", out)

    def test_bench_report_and_import_compare_end_to_end(self):
        srv = FakeServer(("127.0.0.1", 0), ttft_s=0.01, tpot_s=0.002, capacity=8).start()
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg = os.path.join(d, "c.json")
                json.dump({"endpoint": srv.url, "model": "m", "label": "e2e", "metadata": {"gpu_model": "fake", "gpu_count": 1, "gpu_hourly_usd": 1.0},
                           "matrix": {"concurrency": [2], "input_words": [8], "max_tokens": [4]},
                           "repetitions": 3, "requests_per_slot": 2, "warmup_requests": 1, "timeout_s": 10}, open(cfg, "w"))
                out = os.path.join(d, "r.json")
                self.assertEqual(self.run_cli("bench", "--config", cfg, "--out", out)[0], 0)
                code, md = self.run_cli("report", out, "--slo-p95-ms", "5000")
                self.assertEqual(code, 0)
                self.assertIn("# LLM serving benchmark report", md)
                self.assertIn("$/1M out tok", md)
                imp = os.path.join(d, "imp")
                self.run_cli("import-runs", os.path.join(ROOT, "data", "real_gpu_runs.csv"), "--out-dir", imp)
                code, text = self.run_cli("compare", os.path.join(imp, "4090_ep4.json"), os.path.join(imp, "a100_ep4.json"))
                self.assertIn("CONFOUNDED", text)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
