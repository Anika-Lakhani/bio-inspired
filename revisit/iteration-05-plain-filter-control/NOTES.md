# Iteration 5: Plain-filter control — does the center-surround shape matter?

Date run: 2026-11-11

## Purpose

Iteration 4 confirmed that early lateral inhibition beats the Control under
mild noise (+2.9 points at σ=0.05, 10 seeds). This iteration asks whether
the *biological shape* is responsible, or whether any small learnable filter
at the same spot would do as well.

## The control model

A plain 5×5 depthwise filter placed exactly where early inhibition sits
(right after the stem conv, 16 channels at 112×112), with the same strength
parameter α starting at 0 and the same learning rate (1e-2). The only
differences from early inhibition:

| | Early inhibition (iteration 4) | Plain filter (this iteration) |
|---|---|---|
| Starting kernel | Center-surround (DoG) | Random (normal, std 0.1) |
| Zero-sum constraint | Yes | No |
| Sign of α | Forced ≥ 0 | Free |

Same seeds (3–12), training subsets and corrupted test images as iteration
4, so Control and early-inhibition results are reused from there.

## Results (10 seeds, paired by seed)

Plain filter minus Control:

| Condition | Control | Plain filter | Difference | 95% CI | p | Seeds won |
|---|---|---|---|---|---|---|
| Clean | 83.8 ± 1.6 | 82.7 ± 1.9 | −1.0 | [−1.6, −0.4] | 0.004 | 1/10 |
| σ=0.01 | 78.3 ± 2.5 | 79.0 ± 2.8 | +0.7 | [−1.9, +3.3] | 0.58 | 7/10 |
| σ=0.03 | 41.9 ± 4.3 | 44.4 ± 6.0 | +2.5 | [−1.9, +6.8] | 0.23 | 7/10 |
| σ=0.05 | 25.4 ± 3.0 | 28.7 ± 4.1 | +3.2 | [+0.6, +5.9] | 0.021 | 9/10 |
| σ=0.1 | 11.7 ± 1.5 | 12.9 ± 1.3 | +1.2 | [+0.1, +2.4] | 0.040 | 7/10 |
| Occluded | 79.8 ± 2.9 | 79.1 ± 3.5 | −0.7 | [−2.3, +0.9] | 0.33 | 5/10 |

Early inhibition minus plain filter:

| Condition | Difference | 95% CI | p | Seeds won by inhibition |
|---|---|---|---|---|
| Clean | +0.4 | [−0.5, +1.3] | 0.37 | 6/10 |
| σ=0.01 | +1.4 | [−0.7, +3.4] | 0.16 | 7/10 |
| σ=0.03 | −0.3 | [−3.8, +3.3] | 0.86 | 6/10 |
| σ=0.05 | −0.3 | [−3.1, +2.5] | 0.82 | 7/10 |
| σ=0.1 | −0.8 | [−2.0, +0.5] | 0.19 | 4/10 |
| Occluded | −0.7 | [−2.0, +0.5] | 0.19 | 2/10 |

The plain filter's learned α was positive in 6 seeds and negative in 4.

## Findings

1. **The center-surround shape is not what produces the gain.** A plain
   learnable filter at the same spot gives the same benefit at σ=0.05
   (+3.2 vs +2.9) and the two are statistically indistinguishable in every
   condition.
2. **What matters is adding a small, gently-introduced trainable filter at
   the earliest stage.** Both early layers help; late layers (iteration 3)
   don't.
3. **One soft point in inhibition's favor: consistency.** Early inhibition's
   accuracy varied less across seeds under noise (e.g. ±2.4 vs ±6.0 at
   σ=0.03), and it beat the plain filter at σ=0.01 in 7/10 seeds (not
   significant). The biological constraint may make the layer more
   predictable, but this run doesn't establish that.

## What this means for the hypothesis

This is an honest negative for the strong form of the hypothesis. Lateral
inhibition at the first stage helps under mild noise, but no more than an
unconstrained filter in the same place, so the benefit can't be attributed
to the biological mechanism itself. What the iterations do support is that
**where** extra processing goes matters: the earliest stage helps, the final
stage doesn't. That matches where lateral inhibition operates in biology,
but the network gets the same gain without the biological form.

Suggested framing for the paper:

> "An early, retina-like lateral inhibition layer improved robustness to
> mild Gaussian noise (+2.9 points at σ=0.05, 10 seeds). However, an
> unconstrained learnable filter at the same location produced an
> equivalent gain (+3.2 points), indicating that the benefit stems from
> adding adaptable processing at the earliest stage rather than from the
> center-surround structure itself. This mirrors the paper's original
> 'optimization mismatch' hypothesis: standard CNNs can learn what the
> biological prior encodes, given the opportunity."

## Caveats

Same as iteration 4: short training (1 epoch, 20,000 images), Gaussian
noise only, 5,000 test images.
