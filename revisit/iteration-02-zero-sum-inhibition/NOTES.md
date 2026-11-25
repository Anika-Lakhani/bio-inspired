# Iteration 2: Zero-sum center-surround inhibition, starting switched off

Date run: 2026-11-09

## What changed from iteration 1

Only the inhibition layer. Placement (on MobileNetV3's final 576-channel, 7×7
features), training budget, seeds, training subsets and corrupted test images
are identical to iteration 1, so Control and Attention Only are reused from
there.

| | Iteration 1 (paper's layer) | Iteration 2 |
|---|---|---|
| Kernel | Full 576→576 7×7 conv, weights clamped ≤ 0 | One 5×5 kernel per channel (depthwise) |
| Shape | Mixes all channels, all-negative | Difference of Gaussians: excitatory center, inhibitory surround, kept at zero sum |
| New parameters | ~16.6M | 14,401 |
| Strength α at start | 1 (disrupts pretrained features immediately) | 0 (model starts identical to the Control) |
| Learning rate for new parameters | 1e-4 | 1e-2 (at 1e-4, α could move only ~0.03 in one epoch) |

The higher learning rate also applies to the attention/inhibition mixing
weight in the Combined model. The attention module itself is unchanged from
the paper.

## Results (accuracy %, mean ± std over 3 seeds)

| Model          | Clean      | σ=0.01     | σ=0.03     | σ=0.05     | σ=0.1      | Occluded   |
|----------------|------------|------------|------------|------------|------------|------------|
| Control        | 85.0 ± 1.0 | 80.3 ± 1.5 | 44.8 ± 6.0 | 27.0 ± 5.1 | 13.2 ± 1.9 | 81.7 ± 2.1 |
| Attention Only | 85.0 ± 1.4 | 79.2 ± 2.1 | 44.5 ± 6.7 | 31.5 ± 1.2 | 13.9 ± 1.5 | 80.3 ± 5.5 |
| Inhibition v1  | 64.9 ± 1.2 | 62.9 ± 2.9 | 29.1 ± 1.8 | 18.6 ± 3.3 | 14.8 ± 4.4 | 57.4 ± 3.7 |
| Combined v1    | 69.1 ± 3.6 | 66.6 ± 0.7 | 33.9 ± 1.8 | 24.0 ± 1.7 | 11.5 ± 2.3 | 64.1 ± 2.8 |
| **Inhibition v2** | 84.3 ± 1.1 | 79.7 ± 1.7 | 43.8 ± 3.7 | 29.9 ± 6.2 | 15.6 ± 2.2 | 80.6 ± 2.1 |
| **Combined v2**   | 85.1 ± 0.9 | 79.8 ± 1.9 | 44.1 ± 2.8 | 30.3 ± 3.3 | 11.7 ± 1.1 | 80.3 ± 2.7 |

Paired difference from Control, using the same seed and the same corrupted images:

| Model         | Clean      | σ=0.01     | σ=0.03     | σ=0.05     | σ=0.1      | Occluded   |
|---------------|------------|------------|------------|------------|------------|------------|
| Inhibition v2 | −0.6 ± 0.9 | −0.6 ± 1.2 | −1.0 ± 3.7 | +2.9 ± 1.1 | +2.4 ± 3.5 | −1.1 ± 1.7 |
| Combined v2   | +0.1 ± 0.5 | −0.5 ± 1.5 | −0.7 ± 3.9 | +3.3 ± 1.9 | −1.6 ± 2.5 | −1.4 ± 1.3 |

Learned parameters at the end of training:

Note: α alone doesn't show whether the layer sharpens or smooths, because
the kernel's sign is also learned. See the correction under Findings.

| Seed | Inhibition v2 α | Combined v2 α | Combined v2 attention weight |
|---|---|---|---|
| 0 | −0.853 | −1.204 | 0.73 |
| 1 | −0.898 | −1.341 | 0.74 |
| 2 | −0.837 | −1.233 | 0.75 |

## Findings

1. **The redesign removed the harm.** The roughly 20-point clean-accuracy
   penalty from iteration 1 is gone. Both v2 models match the Control on
   clean and occluded images, which answers the "replicate" half of the
   central research question.
2. **There's a small, consistent hint of benefit at moderate noise.** At
   σ=0.05, Inhibition v2 beat the Control in all 3 seeds (+1.6, +3.9, +3.1
   points), and so did Combined v2 (+5.0, +1.3, +3.6). With only 3 seeds this
   is suggestive, not significant (paired t-test for Inhibition v2: p ≈ 0.05).
   There is no benefit at σ=0.01 or σ=0.03, and the result at σ=0.1 is mixed.
3. **The layer did not end up doing lateral inhibition.** In every run, α
   went *negative* (about −0.85 alone, about −1.25 in Combined).

   > **Correction (from iteration 3's kernel check):** This finding
   > originally said that negative α meant the layer had become a smoothing
   > filter. That inference was wrong, because the kernel is learned too
   > and can flip its own sign. Inspecting the effective kernel (α × kernel)
   > for seed 0 shows it drifted to an **unstructured** shape: only 43% of
   > channels center-excitatory, and a mean cosine similarity of +0.03 to
   > the initial center-surround shape. So the late layer was neither
   > lateral inhibition nor clean smoothing. The small σ=0.05 edge can't be
   > credited to either. See
   > `../iteration-03-early-and-nonneg-inhibition/NOTES.md`.

## What this means for the hypothesis

- A well-designed bio-inspired layer no longer hurts accuracy. That's a real
  improvement over both the paper's version and iteration 1.
- The evidence that *lateral inhibition specifically* improves noise
  robustness is not supported here. The layer that showed a hint of benefit
  lost its center-surround structure during training (see the correction
  above).
- ~~A natural paper narrative: "When allowed to learn the sign of its
  center-surround interaction, the network consistently chose suppression of
  contrast rather than enhancement…"~~ Withdrawn after iteration 3's kernel
  check. The late layer's kernel became unstructured rather than inverted.
  Iteration 3 found that the same layer placed *early* keeps its
  center-surround shape and helps under noise. That's the narrative to use.

## Caveats

- Same short budget as iteration 1: 1 epoch on 20,000 images, not converged.
- 3 seeds and 5,000 test images. The σ=0.05 effect needs more seeds before
  it can be claimed.
- Inhibition is placed on 7×7 feature maps, where a 5×5 kernel covers most
  of the map. That's far from where lateral inhibition operates in biology.

## Possible next iterations

- Force α ≥ 0, so the layer can only do real inhibition, to test the
  biological version directly.
- Move inhibition to an early layer (high resolution, low-level features),
  where it makes biological sense.
- Add more seeds to confirm or reject the σ=0.05 effect.
