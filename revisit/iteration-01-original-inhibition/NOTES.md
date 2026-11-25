# Iteration 1: Original inhibition layer, corrected evaluation (failed)

Date run: 2026-11-09

## What this iteration tested

The four models from the paper's notebook (`internal-code.ipynb`), with their
code loaded unchanged. Only the test harness was changed, to fix two bugs
found while reviewing the original project:

1. **Noise clamp bug.** The original code added Gaussian noise *after*
   ImageNet normalization and then clamped to [0, 1]. Normalized pixels span
   about [-2.1, 2.6], so the clamp alone changed 72% of pixel values even
   with zero noise. This is why every model scored ~16% on "noisy" images in
   the paper's Table I. Fixed by adding noise in pixel space, clamping, and
   then normalizing.
2. **Ablation bug.** `ExperimentalCNN` defaults both mechanisms to on, so
   passing only `use_attn=True` still enabled inhibition. "Attention Only",
   "Inhibition Only" and "Combined" in the paper were all the combined model
   (the saved logs print the mixing weight for all three). Fixed by passing
   both flags explicitly.

## Setup

- MobileNetV3-Small pretrained on ImageNet, fine-tuned on CIFAR-10 at 224px
- 1 epoch on a 20,000-image training subset per run (the paper used 5 epochs
  on all 50,000), AdamW lr 1e-4, weight decay 1e-5, gradient clipping 1.0
- 3 seeds (0, 1, 2). Each seed uses its own training subset, and all models
  within a seed see the same subset and the same corrupted test images
- Evaluation on the first 5,000 CIFAR-10 test images: clean, Gaussian noise
  σ ∈ {0.01, 0.03, 0.05, 0.1} in pixel space, and one 50×50 occlusion patch
- Apple M5 GPU (MPS)

## Results (accuracy %, mean ± std over 3 seeds)

| Model           | Clean      | σ=0.01     | σ=0.03     | σ=0.05     | σ=0.1      | Occluded   |
|-----------------|------------|------------|------------|------------|------------|------------|
| Control         | 85.0 ± 1.0 | 80.3 ± 1.5 | 44.8 ± 6.0 | 27.0 ± 5.1 | 13.2 ± 1.9 | 81.7 ± 2.1 |
| Attention Only  | 85.0 ± 1.4 | 79.2 ± 2.1 | 44.5 ± 6.7 | 31.5 ± 1.2 | 13.9 ± 1.5 | 80.3 ± 5.5 |
| Inhibition Only | 64.9 ± 1.2 | 62.9 ± 2.9 | 29.1 ± 1.8 | 18.6 ± 3.3 | 14.8 ± 4.4 | 57.4 ± 3.7 |
| Combined        | 69.1 ± 3.6 | 66.6 ± 0.7 | 33.9 ± 1.8 | 24.0 ± 1.7 | 11.5 ± 2.3 | 64.1 ± 2.8 |

Paired difference from Control, using the same seed and the same corrupted images:

| Model           | Clean       | σ=0.01      | σ=0.03      | σ=0.05     | σ=0.1      | Occluded    |
|-----------------|-------------|-------------|-------------|------------|------------|-------------|
| Attention Only  | +0.0 ± 1.0  | −1.1 ± 1.5  | −0.2 ± 9.1  | +4.5 ± 5.0 | +0.6 ± 2.8 | −1.4 ± 3.7  |
| Inhibition Only | −20.1 ± 0.2 | −17.3 ± 2.4 | −15.7 ± 4.3 | −8.3 ± 8.4 | +1.6 ± 5.1 | −24.2 ± 3.3 |
| Combined        | −15.9 ± 3.4 | −13.7 ± 2.2 | −10.9 ± 5.0 | −3.0 ± 6.8 | −1.7 ± 4.1 | −17.6 ± 2.8 |

## Why we consider it a failure, and what we learned

- **Attention made no measurable difference.** It ties the Control in every
  condition. The +4.5 points at σ=0.05 is within the seed-to-seed spread.
- **Inhibition made the model clearly worse.** It cost ~20 points on clean
  images and ~24 on occluded ones, and it was not more robust relative to its
  own clean accuracy either: at σ=0.03 it kept 45% of its clean accuracy,
  versus 53% for the Control.
- **Likely cause: the layer's design, not the biological idea.** The original
  layer is a full 576→576 7×7 convolution (~16.6M new parameters, about 11×
  the size of the base model) with weights forced to be ≤ 0, and it starts at
  full strength (α = 1). It mixes every channel into every other channel, so
  it isn't center-surround inhibition within a feature map. It also disrupts
  the pretrained features from the very first step, and the network has to
  spend training recovering. In the original 5-epoch logs, the combined model
  recovered to ~92.5% clean (about 1 point below the Control). The 1-epoch
  budget here exposes how much that design slows convergence.
- **Noise sensitivity is real once measured correctly.** The Control drops
  from 85% to 45% at σ=0.03 and to chance at σ=0.1. That leaves plenty of room
  for a mechanism to help, which the clamp bug had hidden.

## Caveat

Training is much shorter than in the paper. These numbers compare the
mechanisms under a fixed small budget and are not converged accuracies.

## Next iteration

Iteration 2 redesigns only the inhibition layer and keeps the placement and
everything else the same. Each channel gets its own zero-sum center-surround
kernel, and the inhibition strength starts at 0, so the model begins
identical to the Control.
