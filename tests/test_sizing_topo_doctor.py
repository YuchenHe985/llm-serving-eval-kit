import os
import unittest

from llmeval import doctor, sizing, topo

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def fx(name):
    with open(os.path.join(FIX, name)) as f:
        return f.read()


class SizingTests(unittest.TestCase):
    def test_kv_bytes_per_token_matches_published_architectures(self):
        self.assertEqual(sizing.kv_bytes_per_token(sizing.PRESETS["llama-3-8b"]), 131072)    # 128 KiB
        self.assertEqual(sizing.kv_bytes_per_token(sizing.PRESETS["llama-3-70b"]), 327680)   # 320 KiB
        self.assertEqual(sizing.kv_bytes_per_token(sizing.PRESETS["qwen2.5-32b"]), 262144)   # 256 KiB

    def test_mla_config_uses_latent_kv(self):
        cfg = {"num_hidden_layers": 61, "num_attention_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
               "hidden_size": 7168}
        spec = sizing.spec_from_hf_config(cfg, 671.0, "mla-like")
        self.assertTrue(spec.mla)
        self.assertEqual(sizing.kv_bytes_per_token(spec), 61 * 576 * 2)  # about 70 KB per token

    def test_gqa_config_and_default_head_dim(self):
        cfg = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}
        spec = sizing.spec_from_hf_config(cfg, 8.03)
        self.assertEqual(spec.kv_elems_per_token_per_layer, 2 * 8 * 128)
        no_gqa = sizing.spec_from_hf_config({"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 64}, 1.0)
        self.assertEqual(no_gqa.kv_heads, 4)  # no num_key_value_heads: plain multi-head attention

    def test_weights_match_the_checkpoint_sizes_seen_in_real_runs(self):
        gib = lambda name: sizing.weights_gib(sizing.PRESETS[name])
        self.assertAlmostEqual(gib("llama-3-8b") * 2**30 / 1e9, 16.06, delta=0.1)      # ~16 GB download
        self.assertAlmostEqual(gib("qwen2.5-32b") * 2**30 / 1e9, 65.0, delta=1.0)      # ~64 GB download
        self.assertAlmostEqual(gib("qwen1.5-moe-a2.7b") * 2**30 / 1e9, 28.6, delta=0.5)  # ~28 GB download
        self.assertEqual(sizing.weights_gib(sizing.PRESETS["llama-3-8b"], "int4") * 4,
                         sizing.weights_gib(sizing.PRESETS["llama-3-8b"], "bf16") * 1)

    def test_estimates_agree_with_the_deployments_that_actually_worked(self):
        gpu4090 = sizing.CATALOG[0]
        kw = dict(concurrency=20, context_tokens=512)
        # 8B fits one 24 GB card: the data-parallel one-replica-per-GPU setup that ran.
        self.assertTrue(sizing.plan(sizing.PRESETS["llama-3-8b"], gpu4090, 1, **kw).fits)
        # 32B needs weights split four ways on 24 GB cards: the TP=4 setup that ran.
        m32 = sizing.PRESETS["qwen2.5-32b"]
        self.assertFalse(sizing.plan(m32, gpu4090, 1, **kw).fits)
        self.assertFalse(sizing.plan(m32, gpu4090, 2, **kw).fits)
        p4 = sizing.plan(m32, gpu4090, 4, **kw)
        self.assertTrue(p4.fits)
        self.assertIn("PCIe", p4.note)  # tensor parallelism without NVLink is flagged
        # ...and does not fit one 40 GB A100 either.
        self.assertFalse(sizing.plan(m32, sizing.CATALOG[2], 1, **kw).fits)
        # MoE (~26.6 GiB of weights) does not fit one 24 GB card.
        self.assertFalse(sizing.plan(sizing.PRESETS["qwen1.5-moe-a2.7b"], gpu4090, 1, **kw).fits)

    def test_recommend_orders_feasible_first_and_max_concurrency_is_consistent(self):
        rec = sizing.recommend(sizing.PRESETS["llama-3-70b"], concurrency=16, context_tokens=4096)
        feasible = [p for p in rec if p.fits]
        self.assertTrue(feasible)
        self.assertEqual(rec[0].fits, True)
        p = feasible[0]
        # Asking for exactly the reported maximum must still fit; one more sequence must not.
        spec = sizing.PRESETS["llama-3-70b"]
        gpu = next(g for g in sizing.CATALOG if g.name == p.gpu)
        self.assertTrue(sizing.plan(spec, gpu, p.tp, concurrency=p.max_concurrency, context_tokens=4096).fits)
        self.assertFalse(sizing.plan(spec, gpu, p.tp, concurrency=p.max_concurrency + 1, context_tokens=4096).fits)

    def test_kv_cache_is_sharded_across_tensor_parallel_ranks_up_to_kv_heads(self):
        m = sizing.PRESETS["qwen2.5-32b"]              # 8 KV heads
        gpu = sizing.CATALOG[4]                        # 80 GiB card so everything fits
        total = 64 * 4096 * sizing.kv_bytes_per_token(m) / 2**30
        self.assertAlmostEqual(sizing.plan(m, gpu, 1, concurrency=64, context_tokens=4096).kv_needed_per_gpu_gib, total)
        self.assertAlmostEqual(sizing.plan(m, gpu, 4, concurrency=64, context_tokens=4096).kv_needed_per_gpu_gib, total / 4)
        # beyond the KV head count the cache is replicated, not split further
        self.assertAlmostEqual(sizing.plan(m, gpu, 16, concurrency=64, context_tokens=4096).kv_needed_per_gpu_gib, total / 8)
        mla = sizing.spec_from_hf_config({"num_hidden_layers": 4, "num_attention_heads": 8, "kv_lora_rank": 32,
                                          "qk_rope_head_dim": 8, "hidden_size": 64}, 1.0)
        t_mla = 10 * 100 * sizing.kv_bytes_per_token(mla) / 2**30
        self.assertAlmostEqual(sizing.plan(mla, gpu, 4, concurrency=10, context_tokens=100).kv_needed_per_gpu_gib, t_mla)

    def test_invalid_tp(self):
        with self.assertRaises(ValueError):
            sizing.plan(sizing.PRESETS["llama-3-8b"], sizing.CATALOG[0], 0)


class TopoTests(unittest.TestCase):
    def test_a100_nvlink(self):
        t = topo.parse_topo(fx("topo_a100_nvlink.txt"))
        self.assertEqual(len(t.gpus), 4)
        self.assertEqual(t.interconnect, "nvlink")
        self.assertEqual(t.links[(0, 1)], "NV12")

    def test_rtx4090_pcie_advises_data_parallel_and_nccl_workarounds(self):
        t = topo.parse_topo(fx("topo_rtx4090_pcie.txt"))
        self.assertEqual(t.interconnect, "pcie")
        advice = " ".join(t.advice())
        self.assertIn("data parallelism", advice)
        self.assertIn("NCCL_P2P_DISABLE", advice)

    def test_mixed_and_errors(self):
        text = "        GPU0 GPU1 GPU2\nGPU0 X NV4 SYS\nGPU1 NV4 X SYS\nGPU2 SYS SYS X\n"
        self.assertEqual(topo.parse_topo(text).interconnect, "mixed")
        with self.assertRaises(ValueError):
            topo.parse_topo("not a topology")
        with self.assertRaises(ValueError):
            topo.parse_topo("GPU0 GPU1\nGPU0 X NV4\n")  # missing row


class DoctorTests(unittest.TestCase):
    def ids(self, name):
        return {f.id for f in doctor.diagnose(fx(name))}

    def test_rtx4090_startup_failures_are_all_recognised(self):
        self.assertEqual(self.ids("log_rtx4090_startup.txt"), {
            "cuda-image-newer-than-driver", "flag-removed-prefix-caching", "nccl-init-failure",
            "flag-renamed-expert-parallel", "multimem-allgather-disabled"})

    def test_a100_startup_failures_are_all_recognised(self):
        self.assertEqual(self.ids("log_a100_startup.txt"), {
            "gpu-oom-at-startup", "gpu-wedged", "gpu-reset-denied", "port-in-use"})

    def test_healthy_log_has_no_findings_and_errors_sort_first(self):
        self.assertEqual(doctor.diagnose(fx("log_healthy.txt")), [])
        sev = [f.severity for f in doctor.diagnose(fx("log_rtx4090_startup.txt"))]
        self.assertEqual(sev, sorted(sev, key=["error", "warning", "info"].index))

    def test_findings_carry_evidence_cause_and_fix(self):
        f = next(x for x in doctor.diagnose(fx("log_rtx4090_startup.txt")) if x.id == "nccl-init-failure")
        self.assertIn("unhandled system error", f.evidence)
        self.assertIn("NCCL_P2P_DISABLE=1", f.fix)
        self.assertIn("ipc: host", f.fix)

    def test_cuda_preflight(self):
        drv = doctor.parse_driver_cuda(fx("nvidia_smi_banner.txt"))
        self.assertEqual(drv, 12.9)
        self.assertIsNotNone(doctor.check_cuda_compat(drv, 13.0))
        self.assertIsNone(doctor.check_cuda_compat(drv, 12.4))
        self.assertIsNone(doctor.parse_driver_cuda("no banner"))

    def test_every_catalog_pattern_compiles_and_ids_are_unique(self):
        import re
        ids = [s.id for s in doctor.CATALOG]
        self.assertEqual(len(ids), len(set(ids)))
        for s in doctor.CATALOG:
            re.compile(s.pattern)
            self.assertTrue(s.cause and s.fix)


if __name__ == "__main__":
    unittest.main()
