# Iteration 8: Neural-noise strength sweep

Date run: 2026-11-11

## Purpose

In iteration 7 the noise strength (0.3) wrecked accuracy, but under that
noise the fixed center-surround prior clearly beat the plain learnable
filter. This iteration tests milder noise strengths, 0.03 and 0.1, to see
whether accuracy recovers and whether the prior's advantage holds.

## The models

There are two models, each run at both strengths:

- **Frozen:** the fixed center-surround filter (α=0.5), followed by
  `NeuralNoise(scale)`.
- **Plain:** the plain learnable filter, followed by `NeuralNoise(scale)`.

As in iteration 7, the noise is on during both training and testing. The
seeds (3–12), training subsets and corrupted images are the same as in
iterations 4–7.

## Results (10 seeds, paired; accuracy points vs Control)

| Condition | Frozen 0.03 | Plain 0.03 | Frozen 0.1 | Plain 0.1 | (Frozen 0.3) | (Plain 0.3) |
|---|---|---|---|---|---|---|
| Clean | −7.0 | −16.9 | −15.6 | −29.6 | −35.9 | −47.9 |
| σ=0.01 | −4.6 | −2.1 (n.s.) | −18.9 | −9.3 | −25.0 | −26.9 |
| σ=0.03 | +1.1 (n.s.) | +3.4 (n.s.) | −6.3 | +4.0 (n.s.) | −9.5 | −10.4 |
| σ=0.05 | +0.6 (n.s.) | +1.3 (n.s.) | +3.2 (p=0.07) | +3.0 (n.s.) | −0.4 | −7.6 |
| Occluded | −6.6 | −20.3 | −16.2 | −30.9 | −33.3 | −44.8 |

**Frozen minus plain**, at the same noise strength:

| Condition | 0.03 | 0.1 | 0.3 |
|---|---|---|---|
| Clean | **+9.9 (10/10 seeds)** | **+14.0 (9/10)** | **+12.1 (9/10)** |
| Occluded | **+13.7 (10/10)** | **+14.7 (9/10)** | **+11.6 (9/10)** |
| σ=0.01 | −2.6 (n.s.) | **−9.6 (0/10)** | +1.9 (n.s.) |
| σ=0.03 | −2.3 (n.s.) | **−10.3 (1/10)** | +0.9 (n.s.) |
| σ=0.05 | −0.8 (n.s.) | +0.2 (n.s.) | **+7.2 (10/10)** |

Seed-to-seed spread on clean images: frozen ±2–3.5, plain ±7–9 at every
strength. Full tables are in `log.txt`.

> **Correction (iteration 9–10):** the frozen-vs-plain advantage below was caused by the plain
> filter's high learning rate (1e-2) and short training. At a normal learning rate, or with 3 epochs of
> training, the plain filter does as well or better. The interpretation that the biological prior
> protects against internal noise is withdrawn.

## Findings

1. **Internal noise never beat the noise-free Control in this setup.** Even
   at 0.03, every noisy model loses 7–17 points on clean images, and no
   noisy model significantly beats the Control under input noise. With one
   epoch of training, noise is a pure cost here. That's unlike VOneNet,
   which trained for far longer.
2. **The fixed biological prior reliably protects the network from internal
   noise.** At every strength, frozen beats plain by 10–15 points on clean
   and occluded images, winning 9–10 of 10 seeds, and its results are 2–3×
   more stable. It's the most robust result in the whole project: it
   replicated at all three strengths.
3. **But the prior doesn't add robustness to external image noise.** Under
   mild input noise at strength 0.1, the plain filter actually does better
   (about 10 points at σ=0.01 and σ=0.03). The prior's benefit is
   tolerance of *internal* noise, not *external* noise.

## What this means for the reframed paper

This is the nature-vs-CNN comparison's clearest result. A learned filter
behaves as well as the biological prior in a clean, deterministic network,
but it falls apart when the network's own units are noisy. The fixed
zero-sum center-surround kernel keeps the network stable. Biology's priors
look adapted to noisy neural hardware, a problem that standard CNNs, being
deterministic, never face.

## Caveats

- Training was short (1 epoch, 20,000 images).
- Noise was on at test time too, so noisy models are judged on noisy
  computation.
- The comparison uses a single prior strength (α=0.5) and a single DoG
  width.
- The plain filter's instability could partly come from its higher
  learning rate (1e-2) interacting with noise.
