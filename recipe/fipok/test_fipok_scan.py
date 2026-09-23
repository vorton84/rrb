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

import sys

import torch

from recipe.fipok.fipok_policy_loss import backward_scan, forward_window_var

def fipo_reference_future_kl(kl_response: torch.Tensor, gamma: float, chunk_size: int = 128):
    batch_size, response_len = kl_response.shape
    device = kl_response.device
    future_kl = torch.zeros((batch_size, response_len), device=device, dtype=kl_response.dtype)
    pos_i = torch.arange(response_len, device=device).unsqueeze(1)
    gamma_t = torch.tensor(gamma, dtype=kl_response.dtype, device=device)
    for j_start in range(0, response_len, chunk_size):
        j_end = min(response_len, j_start + chunk_size)
        j_idx = torch.arange(j_start, j_end, device=device).unsqueeze(0)
        distance = j_idx - pos_i
        mask = distance >= 0
        distance_clamped = distance.clamp(min=0)
        decay_block = torch.pow(gamma_t, distance_clamped) * mask.to(kl_response.dtype)
        kl_block = kl_response[:, j_start:j_end]
        contrib = torch.matmul(kl_block, decay_block.t())
        future_kl += contrib
    return future_kl

def naive_recursion(z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    bsz, seqlen = z.shape
    out = torch.zeros_like(z, dtype=torch.float64)
    z64, g64 = z.double(), gamma.double()
    for t in reversed(range(seqlen)):
        nxt = out[:, t + 1] if t + 1 < seqlen else torch.zeros(bsz, dtype=torch.float64)
        out[:, t] = z64[:, t] + g64[:, t] * nxt
    return out

def naive_window_var(signal, valid, window, fallback):
    bsz, seqlen = signal.shape
    out = torch.zeros(bsz, seqlen, dtype=torch.float64)
    s64 = signal.double()
    for b in range(bsz):
        for t in range(seqlen):
            vals = [s64[b, k].item() for k in range(t + 1, min(t + 1 + window, seqlen)) if valid[b, k] > 0]
            if len(vals) >= 2:
                m = sum(vals) / len(vals)
                out[b, t] = sum((v - m) ** 2 for v in vals) / len(vals)
            else:
                out[b, t] = fallback[b, t].item()
    return out

RESULTS = []

def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

def main():
    torch.manual_seed(0)

    print("\nTest 1 -- constant gain must reproduce FIPO's decay matmul")

    for seqlen, tau in [(512, 32.0), (2048, 32.0), (1024, 128.0), (300, 8.0)]:
        gamma_scalar = 2.0 ** (-1.0 / tau)
        z = torch.randn(4, seqlen) * 1e-3
        ref = fipo_reference_future_kl(z, gamma_scalar)
        got = backward_scan(z, torch.full_like(z, gamma_scalar), chunk_size=64)
        err = (ref - got).abs().max().item()
        scale = ref.abs().max().item()
        check(f"L={seqlen:5d} tau={tau:5.1f}", err < 1e-5 * max(scale, 1e-6) + 1e-9,
              f"max|diff|={err:.3e}  scale={scale:.3e}")

    print("\nTest 2 -- varying gain must satisfy the recursion")

    for seqlen in [7, 64, 200, 513]:
        z = torch.randn(3, seqlen) * 1e-3
        gamma = torch.rand(3, seqlen) * 0.5 + 0.5
        got = backward_scan(z, gamma, chunk_size=64).double()
        ref = naive_recursion(z, gamma)
        err = (ref - got).abs().max().item()
        scale = ref.abs().max().item()
        check(f"L={seqlen:5d} random gamma", err < 1e-5 * max(scale, 1e-6) + 1e-9,
              f"max|diff|={err:.3e}")

    print("\nTest 3 -- result must not depend on chunk size")

    z = torch.randn(2, 777) * 1e-3
    gamma = torch.rand(2, 777) * 0.4 + 0.58
    base = backward_scan(z, gamma, chunk_size=16)
    for cs in [32, 64, 128, 777, 1024]:
        got = backward_scan(z, gamma, chunk_size=cs)
        finite = bool(torch.isfinite(got).all())
        err = (base - got).abs().max().item()
        ok = finite and err < 1e-5 * max(base.abs().max().item(), 1e-6) + 1e-9
        check(f"chunk={cs:5d}", ok, f"finite={finite}  max|diff|={err:.3e}")

    print("\nTest 4 -- forward window variance against a naive loop")

    for window in [4, 16, 64]:
        sig = torch.randn(3, 120) * 1e-3
        valid = (torch.rand(3, 120) > 0.25).float()
        fallback = torch.full((3, 120), 7e-7)
        got = forward_window_var(sig, valid, window, fallback).double()
        ref = naive_window_var(sig, valid, window, fallback)
        err = (ref - got).abs().max().item()
        check(f"W={window:3d}", err < 1e-9, f"max|diff|={err:.3e}")

    print("\nTest 5a -- oversized chunk must be capped, not overflow")

    for seqlen, glo, ghi in [(777, 0.58, 0.98), (4096, 0.5, 0.99), (2048, 0.3, 0.9)]:
        z = torch.randn(2, seqlen) * 1e-3
        gamma = torch.rand(2, seqlen) * (ghi - glo) + glo
        safe = backward_scan(z, gamma, chunk_size=8)
        huge = backward_scan(z, gamma, chunk_size=10 * seqlen)
        finite = bool(torch.isfinite(huge).all())
        err = (safe - huge).abs().max().item()
        ok = finite and err < 1e-5 * max(safe.abs().max().item(), 1e-6) + 1e-9
        check(f"L={seqlen:5d} gamma in [{glo},{ghi}] chunk=10L", ok,
              f"finite={finite}  max|diff|={err:.3e}")

    print("\nTest 5 -- numerical range at the clamp boundaries")

    for gmin, cs in [(0.5, 64), (0.5, 128), (0.3, 64)]:
        z = torch.randn(2, 10240) * 1e-3
        gamma = torch.full_like(z, gmin)
        got = backward_scan(z, gamma, chunk_size=cs)
        finite = bool(torch.isfinite(got).all())
        bound = 1e-3 * 6.0 / (1.0 - gmin)
        sane = got.abs().max().item() < bound
        check(f"gamma_min={gmin} chunk={cs}", finite and sane,
              f"finite={finite}  max|F|={got.abs().max().item():.3e}  bound={bound:.3e}")

    print("\nTest 6 -- adaptive gain reduces to FIPO when the signal is homoscedastic")

    seqlen = 1024
    tau = 32.0
    gamma_target = 2.0 ** (-1.0 / tau)
    z = torch.randn(4, seqlen) * 1e-3
    r_const = torch.full_like(z, 3.3e-7)
    q = gamma_target * 3.3e-7 / (1.0 - gamma_target)
    gamma = (q / (q + r_const)).clamp(0.5, 0.9995)
    got = backward_scan(z, gamma, chunk_size=64)
    ref = fipo_reference_future_kl(z, gamma_target)
    err = (ref - got).abs().max().item()
    check("homoscedastic R -> FIPO", err < 1e-5 * max(ref.abs().max().item(), 1e-6) + 1e-9,
          f"gamma={gamma.mean().item():.6f} target={gamma_target:.6f} max|diff|={err:.3e}")

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
