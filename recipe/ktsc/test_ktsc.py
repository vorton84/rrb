# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import tempfile

from recipe.ktsc.ktsc_controller import KtscConfig, KtscController

RESULTS: list[tuple[str, bool]] = []

def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

def simulate(cfg, c, steps, noise_sigma=0.0, seed=0, spikes=None):
    rng = random.Random(seed)
    ctl = KtscController(cfg)
    mults, kls = [], []
    spikes = spikes or {}
    for t in range(1, steps + 1):
        mult = ctl.lr_mult
        kl = spikes[t] if t in spikes else c * mult * mult * (math.exp(rng.gauss(0.0, noise_sigma)) if noise_sigma > 0 else 1.0)
        ctl.update(kl, global_step=t)
        mults.append(mult)
        kls.append(kl)
    return ctl, mults, kls

def part_a():
    print("\nPart A -- controller")
    d = 1.124e-4

    for label, c in [("10x above", 10 * d), ("10x below", 0.1 * d), ("20x above", 20 * d)]:
        cfg = KtscConfig(delta_target=d, warmup_steps=0)
        ctl, mults, kls = simulate(cfg, c=c, steps=200)
        tail = kls[-20:]
        rel = max(abs(k / d - 1.0) for k in tail)
        target_mult = math.sqrt(d / c)
        check(
            f"A1 converge {label} (median)",
            rel < 0.05 and abs(mults[-1] / target_mult - 1.0) < 0.05,
            f"kl/delta tail max-dev={rel:.3f}  mult={mults[-1]:.4f} target={target_mult:.4f}",
        )
    cfg = KtscConfig(delta_target=d, warmup_steps=0, stat="ema")
    ctl, mults, kls = simulate(cfg, c=10 * d, steps=200)
    check("A1 converge 10x above (ema)", max(abs(k / d - 1.0) for k in kls[-20:]) < 0.05, f"mult={mults[-1]:.4f}")

    cfg = KtscConfig(delta_target=d, warmup_steps=0)
    ctl, mults, kls = simulate(cfg, c=100 * d, steps=200)
    check("A1 100x above pins at mult_min (5x max cut, by design)", mults[-1] == cfg.mult_min and abs(kls[-1] / d - 4.0) < 1e-9,
          f"mult={mults[-1]:.3f} kl/delta={kls[-1] / d:.2f}")

    cfg = KtscConfig(delta_target=d, warmup_steps=0)
    ctl, mults, kls = simulate(cfg, c=8 * d, steps=800, noise_sigma=0.7, seed=1)
    med_tail = statistics.median(kls[-400:])
    check("A2 noisy plant median -> delta", abs(med_tail / d - 1.0) < 0.25, f"median(kl)/delta={med_tail / d:.3f}")

    spikes = {30: 40 * d, 31: 60 * d}
    cfg = KtscConfig(delta_target=d, warmup_steps=0)
    _, m_spk, _ = simulate(cfg, c=1.5 * d, steps=60, spikes=spikes)
    _, m_ref, _ = simulate(cfg, c=1.5 * d, steps=60)
    dev = max(abs(a - b) for a, b in zip(m_spk[29:50], m_ref[29:50]))
    check("A11 median: spikes leave lr trajectory unchanged", dev < 0.02, f"max |mult_spike - mult_ref| = {dev:.4f}")
    cfg = KtscConfig(delta_target=d, warmup_steps=0, stat="ema", ema_beta=0.7, gain=0.5, step_clamp_lo=0.5, mult_min=0.05)
    _, m_spk, _ = simulate(cfg, c=1.5 * d, steps=60, spikes=spikes)
    _, m_ref, _ = simulate(cfg, c=1.5 * d, steps=60)
    dev_v1 = max(abs(a - b) for a, b in zip(m_spk[29:50], m_ref[29:50]))
    check("A11 v1 (ema, old settings) reproduces the collapse", dev_v1 > 0.5 and min(m_spk[29:50]) <= 0.15,
          f"max dev = {dev_v1:.3f}, min mult = {min(m_spk[29:50]):.3f}  (run 12 hit 0.05)")

    cfg = KtscConfig(delta_target=d, warmup_steps=0)
    _, mults, kls = simulate(cfg, c=1.5 * d, steps=120)
    swing = max(mults[-30:]) - min(mults[-30:])
    check("A13 no limit cycle at steady state", swing < 0.02 and abs(mults[-1] - math.sqrt(1 / 1.5)) < 0.02,
          f"swing over last 30 steps = {swing:.4f}, mult = {mults[-1]:.4f} (target 0.8165)")

    cfg = KtscConfig(delta_target=d, warmup_steps=20, seed_steps=5)
    ctl = KtscController(cfg)
    held_ok = True
    for t in range(1, 21):
        ctl.update(1e-9, global_step=t)
        held_ok &= ctl.lr_mult == 1.0 and not ctl.kl_hist
    check("A3 warmup hold", held_ok, f"mult={ctl.lr_mult} hist={ctl.kl_hist} held={ctl.n_held}")
    seed_ok = True
    for i, t in enumerate(range(21, 26)):
        ctl.update((6 + i) * d, global_step=t)
        seed_ok &= ctl.lr_mult == 1.0
    check("A3 seed phase holds mult, fills history", seed_ok and len(ctl.kl_hist) == 5, f"hist/d={[round(x / d, 1) for x in ctl.kl_hist]}")
    ctl.update(10 * d, global_step=26)
    check("A3 first controlled update acts", ctl.lr_mult < 1.0, f"mult={ctl.lr_mult:.4f}")

    cfg = KtscConfig(delta_target=d, warmup_steps=0, seed_steps=5)
    ctl = KtscController(cfg)
    ctl.update(-6.65e-3, global_step=1)
    never_above_one = ctl.lr_mult <= 1.0
    for t in range(2, 30):
        ctl.update(5 * d, global_step=t)
        never_above_one &= ctl.lr_mult <= 1.0
    check("A10 negative first sample never raises lr", never_above_one and ctl.lr_mult < 1.0, f"final mult={ctl.lr_mult:.4f}")

    cfg = KtscConfig(delta_target=d, warmup_steps=0, seed_steps=0, stat="ema", ema_beta=0.0, gain=1.0)
    ctl = KtscController(cfg)
    ctl.update(1e4 * d, global_step=1)
    check("A4 per-step lower clamp", abs(ctl.lr_mult - 0.7) < 1e-12, f"mult={ctl.lr_mult}")
    ctl = KtscController(cfg)
    ctl.update(1e-4 * d, global_step=1)
    check("A4 per-step upper clamp", abs(ctl.lr_mult - 1 / 0.7) < 1e-12, f"mult={ctl.lr_mult}")
    ctl = KtscController(cfg)
    for t in range(1, 40):
        ctl.update(1e4 * d, global_step=t)
    check("A4 global mult_min", ctl.lr_mult == cfg.mult_min, f"mult={ctl.lr_mult}")
    ctl = KtscController(cfg)
    for t in range(1, 40):
        ctl.update(1e-6 * d, global_step=t)
    check("A4 global mult_max", ctl.lr_mult == cfg.mult_max, f"mult={ctl.lr_mult}")

    ctl = KtscController(cfg)
    ctl.update(-6.65e-3, global_step=1)
    ok = math.isfinite(ctl.lr_mult) and 1.0 < ctl.lr_mult <= 1 / 0.7 + 1e-12 and ctl.kl_hist == [0.0]
    check("A5 negative KL treated as 0 and bounded", ok, f"mult={ctl.lr_mult:.4f} hist={ctl.kl_hist}")
    ctl.update(0.0, global_step=2)
    check("A5 zero KL stays finite", math.isfinite(ctl.lr_mult), f"mult={ctl.lr_mult:.4f}")

    cfg = KtscConfig(delta_target=d, adv_aware=True, adv_ref=5.48)
    ctl = KtscController(cfg)
    check("A6 adv_aware tightens for PSO amp1.5", abs(ctl.delta_effective(7.9) / d - 5.48 / 7.9) < 1e-9)
    check("A6 adv_aware no-op for DAPO", ctl.delta_effective(5.48) == d and ctl.delta_effective(3.0) == d)
    check("A6 adv_aware off is a no-op", KtscController(KtscConfig(delta_target=d)).delta_effective(7.9) == d)

    cfg = KtscConfig(delta_target=d, warmup_steps=0, seed_steps=0)
    ctl = KtscController(cfg)
    for t in range(1, 30):
        ctl.update(5 * d * (1 + 0.1 * math.sin(t)), global_step=t)
    sd = json.loads(json.dumps(ctl.state_dict()))
    ctl2 = KtscController(cfg)
    ctl2.load_state_dict(sd)
    ctl.update(5 * d, global_step=30)
    ctl2.update(5 * d, global_step=30)
    check("A7 state round trip", ctl.lr_mult == ctl2.lr_mult and ctl.kl_hist == ctl2.kl_hist)

    cfg_new = KtscConfig(delta_target=1.5e-4, warmup_steps=0, seed_steps=2)
    ctl3 = KtscController(cfg_new)
    ctl3.load_state_dict(sd)
    check("A12 config change resets history, keeps mult", ctl3.lr_mult == ctl.lr_mult and ctl3.kl_hist == [] and ctl3.n_seen == 0)
    ctl3.update(5 * d, global_step=31)
    check("A12 re-seeds after reset (holds)", ctl3.lr_mult == ctl.lr_mult and len(ctl3.kl_hist) == 1)

    cfg = KtscConfig.from_dict({"delta_target": 2e-4, "bogus": 1, "gain": 0.25})
    check("A8 from_dict", cfg.delta_target == 2e-4 and cfg.gain == 0.25 and cfg.stat == "median")
    try:
        KtscConfig.from_dict({"stat": "mean"})
        check("A8 invalid stat rejected", False)
    except AssertionError:
        check("A8 invalid stat rejected", True)

    cfg = KtscConfig(delta_target=d, warmup_steps=0, seed_steps=0)
    ctl = KtscController(cfg)
    ctl.update(3 * d, global_step=1, adv_max=5.48)
    m = ctl.metrics(mult_applied=1.0, adv_max=5.48)
    need = {"actor/ktsc_lr_mult", "actor/ktsc_lr_mult_next", "actor/ktsc_kl_raw", "actor/ktsc_kl_stat", "actor/ktsc_ratio"}
    check("A9 metrics keys", need <= set(m) and all(isinstance(v, float) for v in m.values()), str(sorted(m)))

def part_b():
    print("\nPart B -- k3 estimator")
    try:
        import torch

        from recipe.ktsc.ktsc_actor import k3_kl, ktsc_measure_indices
    except Exception as e:
        print(f"  [SKIP] torch/verl not importable here ({e.__class__.__name__}); run Part B on the pod")
        return

    p = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    q = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    kl_true = float((p * (p / q).log()).sum())
    kl_est = float((p * k3_kl(q.log(), p.log())).sum())
    check("B1 E_p[k3] == KL(p||q)", abs(kl_est - kl_true) < 1e-12, f"{kl_est:.6e} vs {kl_true:.6e}")

    rho = torch.linspace(-3, 3, 601, dtype=torch.float64)
    v = k3_kl(rho, torch.zeros_like(rho))
    check("B2 k3 >= 0 pointwise", bool((v >= -1e-15).all()), f"min={float(v.min()):.2e}")
    small = torch.tensor([1e-3, -1e-3, 1e-2], dtype=torch.float64)
    check("B2 k3 ~ rho^2/2", bool(((k3_kl(small, torch.zeros_like(small)) - 0.5 * small * small).abs() < 1e-6).all()))
    big = torch.tensor([50.0, -50.0])
    check("B3 finite on |rho|=50", bool(torch.isfinite(k3_kl(big, torch.zeros_like(big))).all()))

    idx = ktsc_measure_indices(2048, 4)
    groups = (idx // 32).tolist()
    counts = {g: groups.count(g) for g in set(groups)}
    check("B4 stride=4 hits every group 8x", len(counts) == 64 and set(counts.values()) == {8}, f"{len(counts)} groups")

def part_c():
    print("\nPart C -- trainer mixin wiring")
    try:
        import torch
        from omegaconf import OmegaConf

        from recipe.ktsc.ktsc_trainer import KTSC_STATE_FILE, KtscTrainerMixin
        from verl import DataProto
    except Exception as e:
        print(f"  [SKIP] torch/verl not importable here ({e.__class__.__name__}); run Part C on the pod")
        return

    d = 1.124e-4

    class FakeWG:
        def __init__(self):
            self.seen_meta = []
            self.kl_to_return = 8 * d

        def update_actor(self, batch):
            self.seen_meta.append(dict(batch.meta_info))
            return DataProto(meta_info={"metrics": {"actor/ppo_kl": [self.kl_to_return] * 6, "actor/pg_loss": [0.0]}})

    class Base:
        def __init__(self, tmp):
            self.config = OmegaConf.create(
                {
                    "trainer": {
                        "default_local_dir": tmp,
                        "ktsc": {"delta_target": d, "warmup_steps": 2, "seed_steps": 0, "measure_stride": 4},
                    }
                }
            )
            self.global_steps = 0
            self.actor_rollout_wg = FakeWG()
            self.saved = 0

        def init_workers(self):
            pass

        def _save_checkpoint(self):
            os.makedirs(os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"), exist_ok=True)
            self.saved += 1

        def _load_checkpoint(self):
            pass

    class FakeTrainer(KtscTrainerMixin, Base):
        pass

    with tempfile.TemporaryDirectory() as tmp:
        tr = FakeTrainer(tmp)
        tr.init_workers()
        check("C1 config read from trainer.ktsc", tr.ktsc_cfg.delta_target == d and tr.ktsc_cfg.warmup_steps == 2)
        check("C1 update_actor wrapped", tr.actor_rollout_wg.update_actor == tr._ktsc_update_actor)

        adv = torch.zeros(4, 3)
        adv[1, 2] = -5.48
        mask = torch.ones(4, 3)
        batch = DataProto.from_dict(tensors={"advantages": adv, "response_mask": mask}, meta_info={"temperature": 1.0})

        for step in (1, 2):
            tr.global_steps = step
            out = tr.actor_rollout_wg.update_actor(batch)
        meta = tr.actor_rollout_wg.seen_meta[-1]
        check("C2 meta_info injected", meta.get("ktsc_lr_mult") == 1.0 and meta.get("ktsc_measure_stride") == 4, str(meta))
        check("C2 warmup hold through wrapper", tr.ktsc.lr_mult == 1.0)
        m = out.meta_info["metrics"]
        adv_ok = "actor/ktsc_adv_max" in m and abs(m["actor/ktsc_adv_max"][0] - 5.48) < 1e-5
        check("C2 metrics augmented", "actor/ktsc_lr_mult" in m and adv_ok, str(sorted(m)))

        tr.global_steps = 3
        tr.actor_rollout_wg.update_actor(batch)
        m3 = tr.ktsc.lr_mult
        check("C3 controller reacts to KL above target", 0.69 <= m3 < 1.0, f"mult_next={m3:.4f}")
        tr.global_steps = 4
        tr.actor_rollout_wg.update_actor(batch)
        check("C3 next call carries reduced mult", tr.actor_rollout_wg.seen_meta[-1]["ktsc_lr_mult"] == m3)

        saved_inner = tr._ktsc_orig_update_actor
        tr._ktsc_orig_update_actor = lambda b: DataProto(meta_info={"metrics": {"actor/pg_loss": [0.0]}})
        before = tr.ktsc.lr_mult
        tr.global_steps = 5
        tr.actor_rollout_wg.update_actor(batch)
        check("C4 missing KL metric holds mult", tr.ktsc.lr_mult == before)
        tr._ktsc_orig_update_actor = saved_inner

        tr.global_steps = 10
        tr._save_checkpoint()
        path = os.path.join(tmp, "global_step_10", KTSC_STATE_FILE)
        check("C5 state file written", os.path.isfile(path) and tr.saved == 1, path)
        tr2 = FakeTrainer(tmp)
        tr2.init_workers()
        tr2.global_steps = 10
        tr2._load_checkpoint()
        check("C5 state restored on resume", tr2.ktsc.lr_mult == tr.ktsc.lr_mult and tr2.ktsc.kl_hist == tr.ktsc.kl_hist)

        tr3 = FakeTrainer(tmp)
        tr3.config.trainer.ktsc.enable = False
        tr3.init_workers()
        check("C6 enable=false leaves update_actor untouched", tr3.actor_rollout_wg.update_actor != tr3._ktsc_update_actor)

def main():
    part_a()
    part_b()
    part_c()
    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 60}\n{passed}/{total} passed")
    if passed != total:
        print("FAILED -- do not launch training")
        return 1
    print("All checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
