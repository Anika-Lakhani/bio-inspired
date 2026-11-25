# Iteration 7: VOneNet-style neural noise (failed: noise too strong)

Date run: 2026-11-11

## Purpose

VOneNet (Dapello et al. 2020) got much of its robustness from neural noise
added after its fixed V1 front end. This iteration adds that noise after the
early stage, alone and with the iteration 5 and 6 filters.

## The model

`NeuralNoise(scale=0.3)`: `x + 0.3·sqrt(|x|)·N(0,1)`. The noise is
Poisson-like, growing with the size of the response. As in VOneNet, it is on
during both training and testing. It sits right after the stem conv (16
channels at 112×112). There are three variants:

- **Noise only**
- **Frozen center-surround α=0.5 + noise**
- **Plain learnable filter + noise**

The seeds (3–12), training subsets and corrupted images are the same as in
iterations 4–6.

## Results (10 seeds, paired; difference in accuracy points vs Control)

| Condition | Noise only | Frozen + noise | Plain + noise |
|---|---|---|---|
| Clean | −38.1 | −35.9 | −47.9 |
| σ=0.01 | −25.4 | −25.0 | −26.9 |
| σ=0.03 | −10.0 | −9.5 | −10.4 |
| σ=0.05 | −1.7 (n.s.) | −0.4 (n.s.) | −7.6 |
| σ=0.1 | −0.2 (n.s.) | −0.3 (n.s.) | −1.4 |
| Occluded | −35.0 | −33.3 | −44.8 |

**Frozen + noise vs plain + noise:**

| Condition | Difference | p | Seeds won by frozen |
|---|---|---|---|
| Clean | +12.1 | 0.002 | 9/10 |
| σ=0.05 | +7.2 | 0.002 | 10/10 |
| Occluded | +11.6 | 0.003 | 9/10 |

**Frozen + noise vs noise only:**

| Condition | Difference | p | Seeds won by frozen |
|---|---|---|---|
| Clean | +2.2 | 0.0001 | 10/10 |
| Occluded | +1.7 | 0.0003 | 10/10 |

Full tables are in `log.txt`.

> **Correction (iteration 9–10):** the frozen-vs-plain advantage below was caused by the plain
> filter's high learning rate (1e-2) and short training. At a normal learning rate, or with 3 epochs of
> training, the plain filter does as well or better. The interpretation that the biological prior
> protects against internal noise is withdrawn.

## Findings

1. **At this strength, neural noise is destructive.** All three variants
   lose 35–48 points on clean images and never beat the Control. A noise
   scale of 0.3 swamps the signal at this layer, and one epoch of training
   isn't enough to adapt to it. This is a failed setting, not evidence
   against noise in general.
2. **Under noise, the biological prior clearly beats the plain filter.**
   This is the first iteration where the center-surround shape wins
   significantly over the non-biological control: +12 points clean, +7 at
   σ=0.05, +12 occluded. The plain filter is also much less stable across
   seeds (±9 vs ±3.5). A plausible reason is that a learnable filter can
   amplify the injected noise, while the fixed zero-sum kernel can't.
3. **The fixed prior also helps a little over noise alone** (+2.2 clean,
   +1.7 occluded, 10/10 seeds).

## What this means for the hypothesis

The absolute results are unusable. The relative result is interesting,
though: when the network has to cope with internal noise, a fixed
center-surround prior is significantly more helpful than an unconstrained
filter. That fits the VOneNet picture, where the priors and the noise work
together. Whether it holds at a noise level that doesn't wreck accuracy is
the open question.

## Caveats and next step

Only one noise scale was tried, picked without tuning. Training was short
(1 epoch, 20,000 images). VOneNet calibrated its noise to biological firing
statistics and used a full Gabor V1 bank. The next step is a sweep over
smaller scales (for example 0.03, 0.1), or adding noise only during
training, to find a level where clean accuracy holds up. Then check whether
the frozen > plain advantage survives at that level.
