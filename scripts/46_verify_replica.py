#!/usr/bin/env python3
"""Prove the replica block is causal, permutation-correct and mask-respecting.

Step 2 of DECODER_BUILD.md. Every check is a PERTURBATION test -- change an input that
should not matter and assert the output is bit-identical, or change one that should and
assert it moves. Reading the code and agreeing with it is not evidence; today's three
bugs all passed that bar.

    python scripts/46_verify_replica.py
Run with /scratch/oozak23/envs/track_on_r/bin/python
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.action_decoder import ActionMemory

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make(B=2, T_win=6, M=8, N=12, n_obj=10, D=64, seed=0):
    torch.manual_seed(seed)
    L = (M - 1) + T_win
    robot = torch.randn(B, L, N, D)
    scene = torch.randn(B, L, n_obj, D)
    hist = torch.ones(B, L, dtype=torch.bool)
    pmask = torch.zeros(B, N, dtype=torch.bool)
    pmask[:, -3:] = True                       # last 3 robot slots are padding
    m = ActionMemory(D=D, heads=4, layers=2, M=M, scene_slots=M, n_obj=n_obj).eval()
    return m, robot, scene, hist, pmask, T_win


def main():
    torch.set_grad_enabled(False)
    m, robot, scene, hist, pmask, T_win = make()

    out = m(robot, scene, hist, pmask, T_win)
    check("output shape", out.shape == (robot.shape[0], T_win, robot.shape[2], 64),
          str(tuple(out.shape)))

    # ---- 1. CAUSALITY -----------------------------------------------------------------
    print("\n1. causality: frame j must not depend on anything after it")
    M = m.M
    bad = 0
    for j in range(T_win):
        r2, s2 = robot.clone(), scene.clone()
        t = (M - 1) + j
        r2[:, t + 1:] = torch.randn_like(r2[:, t + 1:]) * 100   # scramble the future
        s2[:, t + 1:] = torch.randn_like(s2[:, t + 1:]) * 100
        o2 = m(r2, s2, hist, pmask, T_win)
        if not torch.equal(out[:, j], o2[:, j]):
            bad += 1
    check("scrambling every frame after t leaves output at t bit-identical", bad == 0,
          f"{bad}/{T_win} frames changed")

    # and the converse -- if the PAST does not matter, the memory is not being used
    r3 = robot.clone()
    r3[:, :M - 1] = torch.randn_like(r3[:, :M - 1])
    o3 = m(r3, scene, hist, pmask, T_win)
    moved = (out[:, -1] - o3[:, -1]).abs().max().item()
    check("changing the past DOES move a later frame (memory is live)", moved > 1e-4,
          f"max delta {moved:.3e}")

    # ---- 2. PADDING -------------------------------------------------------------------
    print("\n2. masking: padded robot slots must not influence real ones")
    r4 = robot.clone()
    r4[:, :, pmask[0]] = torch.randn_like(r4[:, :, pmask[0]]) * 100
    o4 = m(r4, scene, hist, pmask, T_win)
    d = (out[:, :, ~pmask[0]] - o4[:, :, ~pmask[0]]).abs().max().item()
    check("scrambling padded robot slots leaves real points unchanged", d < 1e-5,
          f"max delta {d:.3e}")

    # ---- 3. SCENE MASKING -------------------------------------------------------------
    print("\n3. masking: front-padded history must not influence the output")
    hist2 = hist.clone()
    hist2[:, :4] = False                       # mark the first 4 stream frames padded
    s5 = scene.clone()
    o5a = m(robot, s5, hist2, pmask, T_win)
    s5[:, :4] = torch.randn_like(s5[:, :4]) * 100
    o5b = m(robot, s5, hist2, pmask, T_win)
    d = (o5a - o5b).abs().max().item()
    check("scrambling masked history frames changes nothing", d < 1e-5,
          f"max delta {d:.3e}")

    # ---- 4. SCENE ACTUALLY USED -------------------------------------------------------
    print("\n4. the scene cross-attention is load-bearing")
    s6 = torch.randn_like(scene)
    o6 = m(robot, s6, hist, pmask, T_win)
    d = (out - o6).abs().max().item()
    check("changing the scene changes the output", d > 1e-4, f"max delta {d:.3e}")

    # ---- 5. POINT PERMUTATION ---------------------------------------------------------
    print("\n5. robot point order is an artifact of the query tensor")
    real = (~pmask[0]).nonzero().squeeze(-1)
    perm = real[torch.randperm(len(real))]
    idx = torch.cat([perm, pmask[0].nonzero().squeeze(-1)])
    o7 = m(robot[:, :, idx], scene, hist, pmask, T_win)
    inv = torch.empty_like(idx); inv[idx] = torch.arange(len(idx))
    d = (out - o7[:, :, inv]).abs().max().item()
    check("permuting points permutes the output the same way", d < 1e-4,
          f"max delta {d:.3e}  (self-attention is permutation-equivariant)")

    # ---- 6. DETERMINISM / GRADIENTS ---------------------------------------------------
    print("\n6. sanity")
    check("deterministic in eval", torch.equal(out, m(robot, scene, hist, pmask, T_win)))
    with torch.enable_grad():
        m.train()
        o = m(robot, scene, hist, pmask, T_win)
        o.sum().backward()
        n_none = sum(1 for p in m.parameters() if p.grad is None)
        n_zero = sum(1 for p in m.parameters()
                     if p.grad is not None and p.grad.abs().max() == 0)
    check("every parameter receives gradient", n_none == 0 and n_zero == 0,
          f"{n_none} with no grad, {n_zero} with zero grad")

    n_par = sum(p.numel() for p in m.parameters())
    print(f"\n  {n_par:,} parameters at D=64/2 layers (test config)")
    print(f"\n===== {'ALL CHECKS PASSED' if not FAIL else str(len(FAIL)) + ' FAILED: ' + ', '.join(FAIL)} =====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
