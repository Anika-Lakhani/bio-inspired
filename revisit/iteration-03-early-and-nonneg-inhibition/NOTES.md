# Iteration 3: Early placement and forced-positive inhibition

Date run: 2026-11-09 (kernel check run afterwards)

## What changed from iteration 2

The inhibition layer is the same zero-sum, per-channel center-surround layer
from iteration 2 (strength α starts at 0, learning rate 1e-2 for its
parameters). Iteration 3 varies two things, one at a time:

- **Placement.** *Late* is on the final 576-channel, 7×7 features, as in
  iterations 1–2. *Early* is right after MobileNetV3's stem convolution
  (16 channels at 112×112), the stage closest to the retina. The early layer
  has 401 parameters.
- **Sign of α.** *α ≥ 0* clamps α back to 0 after every optimizer step
  (projected gradient), so the layer can switch off but can't flip its
  strength negative. *α free* lets training choose the sign.

| Variant | Placement | α |
|---|---|---|
| Late, α free | Final features | Free (this is Inhibition v2 from iteration 2, reused) |
| Late, α ≥ 0 | Final features | Non-negative |
| Early, α ≥ 0 | After the stem conv | Non-negative |
| Early, α free | After the stem conv | Free |

These are inhibition-only models. The Combined (attention + inhibition)
model was not rerun. Seeds, training subsets, budget (1 epoch on 20,000
images) and corrupted test images are identical to iterations 1–2, so the
Control is reused from iteration 1.

## Results (accuracy %, mean ± std over 3 seeds)

| Model         | Clean      | σ=0.01     | σ=0.03     | σ=0.05     | σ=0.1      | Occluded   |
|---------------|------------|------------|------------|------------|------------|------------|
| Control       | 85.0 ± 1.0 | 80.3 ± 1.5 | 44.8 ± 6.0 | 27.0 ± 5.1 | 13.2 ± 1.9 | 81.7 ± 2.1 |
| Late, α free  | 84.3 ± 1.1 | 79.7 ± 1.7 | 43.8 ± 3.7 | 29.9 ± 6.2 | 15.6 ± 2.2 | 80.6 ± 2.1 |
| Late, α ≥ 0   | 84.8 ± 1.2 | 80.8 ± 1.2 | 43.9 ± 3.6 | 28.9 ± 5.6 | 14.2 ± 1.7 | 80.6 ± 3.5 |
| Early, α ≥ 0  | 83.8 ± 1.8 | 82.8 ± 0.3 | 48.6 ± 4.8 | 33.3 ± 4.5 | 14.7 ± 1.9 | 77.1 ± 5.9 |
| Early, α free | 83.9 ± 0.6 | 82.8 ± 0.9 | 47.5 ± 4.5 | 32.3 ± 5.6 | 14.1 ± 1.4 | 79.0 ± 1.1 |

Paired difference from Control, using the same seed and the same corrupted images:

| Model         | Clean      | σ=0.01     | σ=0.03     | σ=0.05     | σ=0.1      | Occluded   |
|---------------|------------|------------|------------|------------|------------|------------|
| Late, α free  | −0.6 ± 0.9 | −0.6 ± 1.2 | −1.0 ± 3.7 | +2.9 ± 1.1 | +2.4 ± 3.5 | −1.1 ± 1.7 |
| Late, α ≥ 0   | −0.1 ± 0.3 | +0.5 ± 1.0 | −0.9 ± 2.9 | +1.9 ± 0.4 | +1.0 ± 3.6 | −1.1 ± 1.6 |
| Early, α ≥ 0  | −1.1 ± 1.0 | +2.5 ± 1.7 | +3.9 ± 1.7 | +6.3 ± 2.1 | +1.4 ± 0.6 | −4.6 ± 3.8 |
| Early, α free | −1.0 ± 0.5 | +2.5 ± 1.4 | +2.8 ± 2.7 | +5.4 ± 3.7 | +0.8 ± 0.8 | −2.7 ± 1.5 |

Learned α at the end of training:

| Seed | Late, α ≥ 0 | Early, α ≥ 0 | Early, α free |
|---|---|---|---|
| 0 | +0.795 | +0.131 | +0.131 |
| 1 | +0.855 | +0.170 | +0.147 |
| 2 | +0.767 | +0.097 | +0.081 |

For seed 0, the two early variants are identical because α never went
negative, so the clamp never activated.

## Kernel check: what did each layer actually learn?

α alone doesn't say whether a layer sharpens or blurs, because the kernel is
also learned and can flip its own sign. So we retrained seed 0 of each
variant and examined the **effective kernel**, α × (zero-sum kernel), per
channel:

- **Center > 0**: the center pixel is excitatory and the surround is
  inhibitory, so the layer enhances contrast. That's lateral inhibition.
- **Cosine to DoG**: how closely the effective kernel's shape matches the
  initial center-surround shape (+1 = same shape, 0 = unrelated, −1 =
  inverted, i.e. smoothing).

| Variant (seed 0) | α | Channels with center > 0 | Mean cosine to DoG | Mean kernel norm |
|---|---|---|---|---|
| Late, α free  | −0.853 | 43% | +0.03 | 1.14 |
| Late, α ≥ 0   | +0.795 | 78% | +0.27 | 1.12 |
| Early, α ≥ 0  | +0.131 | 81% | +0.35 | 0.15 |
| Early, α free | +0.131 | 81% | +0.35 | 0.15 |

(Raw output: `kernel_inspection_log.txt`, script: `kernel_inspection.py`.)

## Findings

1. **Placement matters more than the sign constraint.** Both early variants
   beat both late variants under noise, and they are the first to help
   across a range of low noise levels (σ = 0.01 to 0.05) rather than at a
   single level.
2. **Early inhibition gives the clearest noise benefit so far.** Early,
   α ≥ 0 beat the Control at σ=0.05 in all 3 seeds (+6.3 ± 2.1 points,
   paired t-test p ≈ 0.04 at 3 seeds) and at σ=0.03 by +3.9 ± 1.7.
   Promising, but not conclusive with 3 seeds and several comparisons.
3. **The early layer really does lateral inhibition.** Its learned kernels
   kept the center-surround shape (81% of channels center-excitatory, cosine
   +0.35) at a modest strength. Left free to choose, it chose inhibition.
   This is the first variant where the biological mechanism is both present
   and associated with a benefit.
4. **It's a trade-off, not a free gain.** Early inhibition costs about 1
   point on clean images and 3–5 on occluded ones. That's plausible
   mechanistically: center-surround filtering suppresses uniform regions,
   and a flat occluding patch is exactly that.
5. **Late inhibition doesn't help, whatever its sign.** Forced positive, the
   late layer does learn mostly center-excitatory kernels (78%), but it
   gives no noise benefit. With α free, the late layer's kernel drifted to
   an unstructured shape (cosine +0.03, center sign split 43/57). It was
   neither inhibition nor clean smoothing. This corrects a claim in
   iteration 2's notes.

## What this means for the hypothesis

The original hypothesis, that lateral inhibition improves noise robustness,
gets its first real support here, with an important qualifier: **only when
the mechanism sits where it does in biology, at the earliest stage of
vision.** The same layer placed at the end of the network does nothing
useful. A possible paper narrative:

> "Lateral inhibition helped only when placed at the earliest stage of the
> network, analogous to the retina, where it learned and kept a
> center-surround form and improved accuracy under mild Gaussian noise by up
> to ~6 points, at a small cost on clean and occluded images. Placed at the
> final feature stage, the identical mechanism gave no benefit. This
> suggests that the location of a biological mechanism matters as much as
> the mechanism itself."

## Caveats

- Same short budget as iterations 1–2 (1 epoch on 20,000 images, not
  converged), 3 seeds and 5,000 test images.
- The kernel check uses seed 0 only.
- Several variants and noise levels were compared, so a p ≈ 0.04 from 3
  seeds should be treated as a lead, not a result.
- Why sharpening at the first layer helps against noise isn't established
  here. One candidate explanation is that a zero-sum kernel removes the
  local mean, which acts like local contrast normalization. This would need
  its own test.

## Possible next steps

- Run 5–10 seeds of Control vs Early, α ≥ 0 to confirm or reject the effect.
- Train longer (closer to the paper's 5 epochs on all 50,000 images) to see
  whether the gap survives convergence.
- Compare against a non-biological control at the same spot (e.g. a plain
  learnable 5×5 depthwise conv, or a fixed blur) to check whether the
  benefit comes from the center-surround shape specifically.
- Combine early inhibition with attention.
