# Iteration 4: 10-seed confirmation of early lateral inhibition

Date run: 2026-11-11

## Purpose

Iteration 3's best variant was early inhibition with α ≥ 0: a zero-sum,
per-channel center-surround layer right after the stem convolution. Over 3
seeds it gave +6.3 ± 2.1 points at σ=0.05. Those 3 seeds were also used to
pick that variant as the winner, so this run checks it on **10 fresh seeds
(3–12)**. Fresh seeds mean different training subsets and different
corrupted test images.

The analysis was fixed before running: σ=0.05 as the primary test, σ=0.03
and the occlusion cost as secondary checks. Model code, training budget
(1 epoch on 20,000 images) and evaluation (first 5,000 test images) are
unchanged from iterations 1–3.

## Results

Early, α ≥ 0 minus Control, paired by seed (each pair saw the same
corrupted test images):

| Condition | Control | Early inhibition | Difference | 95% CI | p (paired t) | Seeds won |
|---|---|---|---|---|---|---|
| Clean | 83.8 ± 1.6 | 83.1 ± 1.3 | −0.7 | [−1.3, −0.0] | 0.042 | 3/10 |
| σ=0.01 | 78.3 ± 2.5 | 80.3 ± 1.7 | +2.0 | [+0.1, +4.0] | 0.040 | 8/10 |
| σ=0.03 | 41.9 ± 4.3 | 44.1 ± 2.4 | +2.2 | [−1.0, +5.4] | 0.156 | 6/10 |
| **σ=0.05 (primary)** | 25.4 ± 3.0 | 28.4 ± 3.0 | **+2.9** | **[+0.8, +5.1]** | **0.012** | **9/10** |
| σ=0.1 | 11.7 ± 1.5 | 12.2 ± 0.9 | +0.5 | [−0.5, +1.5] | 0.308 | 6/10 |
| Occluded | 79.8 ± 2.9 | 78.3 ± 3.2 | −1.5 | [−3.4, +0.5] | 0.122 | 3/10 |

Learned α was positive in every seed (+0.04 to +0.12), so the layer stayed
switched on as inhibition throughout.

## Findings

1. **The primary effect replicates, at about half the size.** On fresh
   seeds, early inhibition beat the Control at σ=0.05 by +2.9 points
   (95% CI +0.8 to +5.1, p = 0.012), winning 9 of 10 seeds. Iteration 3's
   +6.3 was an overestimate, as is typical when the same small sample picks
   the winner and measures it.
2. **The benefit is consistent at mild noise and absent at heavy noise.**
   It was positive at σ=0.01 (+2.0, p = 0.04) and σ=0.03 (+2.2, not
   significant). At σ=0.1 both models are near chance, so there's nothing
   to gain.
3. **There is a small, real cost on clean images.** −0.7 points
   (p = 0.04). The occlusion cost seen in iteration 3 (−4.6) shrank to −1.5
   and is no longer significant.
4. **Multiple comparisons.** σ=0.05 was declared the primary test before
   the run, so its p = 0.012 stands on its own as the confirmatory result.
   No condition would survive a Bonferroni correction across all six
   (σ=0.05 would become p ≈ 0.07). So the σ=0.01 and clean-image results
   should be reported as supporting evidence, not confirmatory.

## What this means for the hypothesis

The hypothesis gets modest but real support, with an important condition:

> Lateral inhibition, implemented as a zero-sum center-surround filter at the
> earliest stage of the network, improved accuracy under mild Gaussian noise
> (+2.9 points at σ=0.05, 95% CI +0.8 to +5.1, 9/10 seeds) at a cost of
> 0.7 points on clean images. The identical mechanism placed at the final
> feature stage gave no benefit (iteration 3).

Combined with iterations 1–3, the overall narrative for the paper:

1. The original results were artifacts of two bugs (noise clamp, ablation
   flags).
2. With those fixed, the original mechanisms either did nothing (attention)
   or hurt badly (the original inhibition layer).
3. Redesigning inhibition to be biologically faithful (per-channel
   center-surround, starting switched off) removed the harm.
4. Moving it to the retina-like first stage produced a small, replicated
   robustness gain, and the learned kernels kept their center-surround
   shape there.

## Caveats

- Short training (1 epoch on 20,000 images), not converged. The effect
  could grow, shrink or vanish with full training.
- Only Gaussian noise was tested. Other corruptions (blur, JPEG, contrast)
  are untested.
- It hasn't been shown that the *center-surround shape* is what matters,
  rather than any small learnable filter at that spot. The natural next
  control is a plain learnable 5×5 depthwise conv at the same position.
