# Iteration 10: Longer training (3 epochs)

Date run: 2026-11-12

## Purpose

Every earlier run trained for one epoch. This iteration checks whether the
frozen-vs-plain comparison under internal noise (0.03) holds with 3 epochs.
It runs the Control, Frozen DoG + noise and Plain (lr 1e-2) + noise over 5
seeds (3–7), with a new shuffle each epoch.

## Results (accuracy points, 5 seeds)

| Condition | Control | Frozen + noise | Plain + noise | Frozen − Plain |
|---|---|---|---|---|
| Clean | 91.0 | 88.9 | 90.0 | −1.1 (0/5) |
| σ=0.01 | 43.3 | 56.2 | 65.7 ± 11 | −9.5 (n.s.) |
| σ=0.03 | 9.4 | 11.0 | 12.1 | −1.2 (n.s.) |
| Occluded | 89.0 | 85.0 | 86.9 | −1.9 (0/5) |

## Findings

1. **With longer training, the plain filter catches up and passes the
   frozen prior** on clean and occluded images. The iteration 7–8 advantage
   was a short-training effect, which matches iteration 9's
   learning-rate diagnosis.
2. **Longer training makes the noise-free Control extremely fragile.** Its
   clean accuracy rose to 91%, but it falls to 43% at σ=0.01 and to chance
   at σ=0.03. Training with internal noise recovers 13–22 points at
   σ=0.01, but neither noisy model holds up at σ≥0.03. The models are
   specializing to clean images.
3. **The cost of internal noise on clean images shrinks with training,**
   from −7 at 1 epoch to −1 or −2 at 3 epochs.

## Caveats

There are only 5 seeds. The learnable center-surround from iteration 9 was
not included, so that is the obvious next comparison.
