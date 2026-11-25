# Iteration 11: Learnable center-surround vs plain filter at a matched, moderate learning rate

Date run: 2026-11-14

## Purpose

At a learning rate of 1e-2 (shared by both layers), the learnable
center-surround layer trained with internal noise 0.03 beat the plain
filter by +8.8 points on clean images and +10.5 on occluded ones (10/10
seeds each). It also gave the project's largest input-noise gain vs the
Control (iteration 9). This iteration checks whether that holds at a
moderate learning rate (1e-3) for both layers. It uses 10 seeds (3–12) and
the Control from iteration 4.

The Python packages were reinstalled before this run (same torch and
torchvision versions) after temp-folder cleanup damaged the environment.

## Results (accuracy points, 10 seeds, paired)

Center-surround minus plain filter, by learning rate:

| Condition | Both at 1e-2 (iters 8–9) | p | Both at 1e-3 (this run) | p | Seeds won (1e-3) |
|---|---|---|---|---|---|
| Clean | **+8.8** | 0.001 | +0.8 | 0.29 | 5/10 |
| σ=0.01 | +2.6 | 0.14 | **+3.1** | 0.004 | 9/10 |
| σ=0.03 | +6.5 | 0.11 | +0.8 | 0.70 | 7/10 |
| σ=0.05 | +6.3 | 0.11 | +2.9 | 0.07 | 8/10 |
| Occluded | **+10.5** | 0.0005 | −0.1 | 0.90 | 4/10 |

Vs the Control at 1e-3, both layers lose 6.5–7.5 points on clean images,
and neither beats the Control under input noise (the center-surround
layer's gains are +1.4 and +0.9, both n.s.). The center-surround α settled
at +0.15 to +0.17 in every seed. The plain filter's α ranged from −0.16 to
+0.15, with no consistent sign.

## Findings

1. **At a moderate learning rate, the shape mostly stops mattering.** The
   large 1e-2 advantages on clean and occluded images disappear. A small,
   fairly consistent edge remains under mild input noise (+3.1 at σ=0.01,
   9/10 seeds, p=0.004; +2.9 at σ=0.05, 8/10, p=0.07).
2. **The big gains need fast learning.** The +10 input-noise gain vs the
   Control (iteration 9) only appears at 1e-2, where α grows to 0.2–0.56.
   At 1e-3, α stays near 0.16 and the layer barely engages.
3. **What the biological shape reliably does is stabilize.** At 1e-2 the
   plain filter becomes erratic under noise (±7–9 spread), while the
   center-surround stays stable (±3.6). The center-surround also converges
   to the same strength in every seed, whereas the plain filter's strength
   and sign vary. The zero-sum, center-surround constraint acts as a guard
   rail that lets the layer learn aggressively without breaking.

## What this means for the paper

The result to write up is a narrow, defensible one. The biological shape
isn't a better filter. It's a safer one: a constraint that keeps a
fast-adapting early layer stable in a noisy network, which is where it
produced the study's largest robustness gains. That's a modest but genuine
nature-vs-CNN finding.

## Caveats

1 epoch only, and the noise stays on at test time. The learning-rate range
was tested at two points only (1e-2 and 1e-3).
