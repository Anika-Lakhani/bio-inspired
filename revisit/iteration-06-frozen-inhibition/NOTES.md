# Iteration 6: Frozen (fixed) lateral inhibition, VOneNet-style

Date run: 2026-11-11

## Purpose

Iteration 5 showed that learnable early inhibition is no better than a plain
learnable filter at the same spot. Published biologically inspired front
ends that improved robustness (e.g. VOneNet, Dapello et al. 2020) mostly use
**fixed**, untrained biological filters. This iteration tests whether
imposing the center-surround prior, so training can't reshape or switch it
off, beats the plain filter.

## The model

The same zero-sum 5×5 center-surround (DoG) kernel, per channel, right after
the stem conv (16 channels at 112×112). Kernel and strength α are stored as
fixed buffers: nothing in the layer trains. Two strengths:

- **α = 0.1:** about what the learnable early layer chose (+0.04 to +0.17)
- **α = 0.5:** a strong prior

Same seeds (3–12), training subsets and corrupted images as iterations 4–5.

## Results (10 seeds, paired by seed; difference in accuracy points)

Versus the **Control**:

| Condition | Frozen α=0.1 | p | Frozen α=0.5 | p | (Learnable, it. 4) | (Plain filter, it. 5) |
|---|---|---|---|---|---|---|
| Clean | −0.6 | 0.09 | −1.0 | 0.010 | −0.7 | −1.0 |
| σ=0.01 | +0.5 | 0.39 | +1.4 | 0.046 | +2.0 | +0.7 |
| σ=0.03 | +1.1 | 0.27 | +2.4 | 0.019 | +2.2 | +2.5 |
| σ=0.05 | +1.4 | 0.15 | **+2.7 (10/10 seeds)** | **0.002** | +2.9 | +3.2 |
| σ=0.1 | +0.5 | 0.25 | +0.2 | 0.58 | +0.5 | +1.2 |
| Occluded | −0.3 | 0.70 | **−4.0** | 0.010 | −1.5 | −0.7 |

Versus the **plain filter** (the key test):

| Condition | Frozen α=0.1 | p | Frozen α=0.5 | p |
|---|---|---|---|---|
| Clean | +0.5 | 0.11 | +0.0 | 0.98 |
| σ=0.01 | −0.2 | 0.86 | +0.7 | 0.43 |
| σ=0.03 | −1.4 | 0.41 | −0.1 | 0.96 |
| σ=0.05 | −1.9 | 0.23 | −0.6 | 0.65 |
| σ=0.1 | −0.8 | 0.27 | −1.1 | 0.11 |
| Occluded | +0.4 | 0.33 | −3.3 | 0.032 |

Full tables, including frozen vs learnable inhibition, are in `log.txt`.

## Findings

1. **Freezing the prior doesn't beat the plain filter.** Neither strength is
   significantly better than an unconstrained learnable filter under noise.
   The VOneNet-style "impose, don't learn" idea doesn't rescue the
   center-surround shape in this setup.
2. **A strong fixed prior gives the most *consistent* noise gain.** Frozen
   α=0.5 beat the Control at σ=0.05 in all 10 seeds (p = 0.002), and at
   σ=0.01 and σ=0.03 (both p < 0.05), with the lowest seed-to-seed spread
   of any variant. But its average gain is no larger than the learnable
   layers'.
3. **The strong prior has a clear cost on occlusion.** −4.0 points vs the
   Control and −3.3 vs the plain filter. That's consistent with
   center-surround filtering suppressing uniform regions, which is what an
   occluding patch is.
4. **A weak fixed prior is too weak.** Frozen α=0.1 helps less than the
   learnable layer, which also settles near 0.1 but adapts its kernel
   shape. Some adaptation is worth having.

## What this means for the hypothesis

Across iterations 3–6, every early-stage layer gives about the same
robustness gain (+1.4 to +3.2 points at σ=0.05), whether it's biological or
not, learned or fixed. The center-surround form adds consistency and a
trade-off (better under noise, worse under occlusion), but not extra
robustness.

A defensible conclusion for the paper:

> "Adding a lightweight filtering stage at the network's earliest layer
> reliably improved robustness to mild Gaussian noise (+1.4 to +3.2 points
> at σ=0.05 across variants). Whether the filter was biologically shaped,
> unconstrained, learned or fixed made no significant difference to the
> size of the gain. A strong fixed center-surround prior made the gain the
> most consistent (10/10 seeds) at the cost of occlusion robustness,
> mirroring the trade-off expected from biological lateral inhibition."

## Caveats

Same as iterations 4–5: short training (1 epoch, 20,000 images), Gaussian
noise only, 5,000 test images. Only two fixed strengths were tried, and
the DoG width wasn't tuned. VOneNet's reported gains relied heavily on
injected neural noise and a fuller V1 model (Gabor filters, simple/complex
cells), which this iteration doesn't include.
