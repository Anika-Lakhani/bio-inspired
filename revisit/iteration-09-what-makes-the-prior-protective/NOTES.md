# Iteration 9: What makes the prior protective? (Answer: mostly a learning-rate artifact)

Date run: 2026-11-12

## Purpose

In iterations 7–8, the frozen center-surround prior beat the plain learnable
filter by 10–15 points under internal neural noise. This iteration tests
three explanations, all at noise 0.03, 1 epoch and 10 seeds (3–12). It
reuses the Control from iteration 4 and Frozen/Plain at 0.03 from
iteration 8.

| Variant | Question it answers |
|---|---|
| Plain filter, normal learning rate (1e-4, not 1e-2) | Was the plain filter's collapse caused by its high learning rate? |
| Learnable zero-sum center-surround (α ≥ 0) + noise | Does the shape protect even when it's learned? |
| Fixed blur (DoG center only, not zero-sum) + noise | Is it the inhibitory surround, or just smoothing? |

## Results (accuracy points)

| Condition | Frozen DoG vs plain at normal LR | Plain normal LR vs plain at high LR | Learned DoG vs Control | Blur vs Control | Frozen DoG vs blur |
|---|---|---|---|---|---|
| Clean | −0.2 (n.s.) | **+10.1** | −8.1 | −6.2 | −0.8 |
| σ=0.01 | +0.7 (n.s.) | −3.3 | +0.5 (n.s.) | −1.9 | −2.7 |
| σ=0.03 | −0.6 (n.s.) | −1.7 (n.s.) | **+9.9 (9/10, p=0.002)** | **+5.1** | −4.0 |
| σ=0.05 | +2.7 (p=0.03) | −3.5 (n.s.) | **+7.6 (9/10, p=0.007)** | **+5.3** | −4.7 |
| Occluded | +0.2 (n.s.) | **+13.5** | −9.9 | −6.0 | −0.6 |

The learnable DoG layer's α grew to between +0.22 and +0.56 in every seed.
That's 2–10× larger than it grew without noise (iteration 4), so internal
noise pushes the network to use the inhibition layer.

## Findings

1. **Iterations 7–8's headline result was a confound.** With a normal
   learning rate, the plain filter is as stable and accurate as the frozen
   prior: 76.9 vs 76.8 on clean images, ±1.8 spread. Its collapse came from
   the high learning rate (1e-2) interacting with noise, not from lacking a
   biological shape. That claim is withdrawn.
2. **Frozen center-surround is not special.** Under noise, a fixed plain
   blur is actually *better* than the fixed DoG on input noise (+4 to +5
   points at σ=0.03–0.05) and the same on clean images.
3. **New, positive result: a learnable center-surround layer trained with
   internal noise gives the biggest input-noise robustness of any variant
   in the project.** It gets +9.9 at σ=0.03 and +7.6 at σ=0.05 vs the
   Control (9/10 seeds each), at a cost of −8 on clean and −10 on occluded.
   That's roughly 3× the best noise-free gain (iterations 4–5, about +3). It
   also beats the frozen DoG by 7–9 points under input noise. It has not yet
   been compared with a plain filter trained with noise at a matched
   learning rate.

## Caveats

1 epoch only, and the noise stays on at test time. The learnable DoG used
lr 1e-2 for its parameters (as in iteration 4), while the plain filter it
should be compared with ran at 1e-4.
