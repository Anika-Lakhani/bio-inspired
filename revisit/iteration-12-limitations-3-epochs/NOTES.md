# Iteration 12: Final limitations run (3 epochs, 3 learning rates, noise on vs off at test)

Date run: 2026-11-14 to 2026-11-17

## Purpose

This run addresses the three main limitations of iterations 1–11:

1. **Short training:** every model now trains for 3 epochs on the
   20,000-image subset, with a new shuffle each epoch.
2. **Learning rate tested at only two points:** the learnable
   center-surround (CS) layer and the plain filter are each trained at three
   learning rates, 1e-2, 3e-3 and 1e-3.
3. **Internal noise on at test time:** every model is scored twice, with
   internal noise ON (as trained) and OFF.

All models use internal noise 0.03 right after the stem. The Control is
re-run in the same environment. 7 models × 5 seeds (3–7), paired by seed,
with the same corrupted images as before.

## Results (accuracy %, mean ± sd over 5 seeds; noise ON at test)

| Model | Clean | σ=0.01 | σ=0.03 | σ=0.05 | Occluded |
|---|---|---|---|---|---|
| Control (no noise) | 91.0 ± 0.1 | 43.3 ± 2.2 | 9.4 | 9.8 | 89.0 |
| CS, lr 1e-2 | 90.3 ± 0.2 | **74.1 ± 7.2** | 16.9 ± 7.5 | 10.1 | 87.5 |
| Plain, lr 1e-2 | 90.0 ± 0.3 | 65.7 ± 11.2 | 12.1 ± 3.0 | 10.7 | 86.9 |
| CS, lr 3e-3 | 89.8 | 64.2 ± 10.4 | 12.4 | 9.3 | 86.2 |
| Plain, lr 3e-3 | 89.6 | 64.9 ± 4.6 | 11.6 | 10.2 | 86.0 |
| CS, lr 1e-3 | 89.3 | 57.3 ± 7.5 | 12.1 | 9.8 | 85.5 |
| Plain, lr 1e-3 | 89.3 | 61.6 ± 5.3 | 12.7 | 9.6 | 85.4 |

**CS minus plain filter** at the same learning rate (noise ON):

| Condition | lr 1e-2 | lr 3e-3 | lr 1e-3 |
|---|---|---|---|
| Clean | +0.3 (n.s.) | +0.2 (n.s.) | +0.1 (n.s.) |
| σ=0.01 | **+8.4 (5/5, p=0.02)** | −0.7 (n.s.) | −4.3 (n.s.) |
| σ=0.03 | +4.7 (n.s.) | +0.8 (n.s.) | −0.6 (n.s.) |
| Occluded | +0.5 (n.s.) | +0.2 (n.s.) | +0.1 (n.s.) |

**Learned layer strength (α) across seeds:**

| Learning rate | CS α | Plain α |
|---|---|---|
| 1e-2 | +0.44 to +0.67 | ±1.1 to ±1.5, sign flips between seeds |
| 3e-3 | +0.47 to +0.62 | ±0.61 to ±0.65, sign flips |
| 1e-3 | +0.28 to +0.32 | ±0.34 to ±0.38, sign flips |

**Turning internal noise OFF at test time** costs clean accuracy:

| Learning rate | Clean-image cost |
|---|---|
| 1e-2 | −10 to −11 |
| 3e-3 | −25 |
| 1e-3 | −45 to −55 |

The σ=0.01 gains survive mostly intact, except CS at 1e-3, which loses
its gain.

## Findings

1. **With longer training, internal noise becomes nearly free and very
   protective.** At 3 epochs, training with noise costs only 0.6–1.7 points
   on clean images. In return it gains +14 to +31 points at σ=0.01, where
   the noise-free Control collapses to 43%. This is the largest and most
   robust effect in the project, and it holds for every layer type and
   learning rate. At 1 epoch, the clean cost was about 7 points. Under
   stronger image noise (σ ≥ 0.05) every model is at chance, so the
   protection only covers mild corruption.
2. **The CS "guard rail" result replicates at 3 epochs, and only at the
   fast learning rate.** At lr 1e-2, CS beats the plain filter by +8.4 at
   σ=0.01 (5/5 seeds), and CS at 1e-2 is the most noise-robust model in the
   whole study (74.1% vs the Control's 43.3%). At slower rates the two tie.
   CS always learns a consistent positive strength. The plain filter's
   strength flips sign from seed to seed and grows to ±1.5 at lr 1e-2.
   That's the same pattern seen at 1 epoch (iterations 9 and 11).
3. **The networks come to rely on the noise.** Removing it at test time
   hurts clean accuracy (−10 to −55 points), so noise works as a permanent
   part of the computation, not a removable training trick. That matches
   VOneNet, which keeps its noise on at test time, and real neurons, which
   are never noise-free. The dependence is strongest at slow learning
   rates, where the rest of the network has adapted around a weakly
   engaged layer. Noise ON is the right way to evaluate these models.

## Limitations that remain

- Only 5 seeds, 3 epochs, a 20,000-image subset and CIFAR-10 at 224px with
  one backbone (MobileNetV3-Small).
- One internal-noise level (0.03).
- Gaussian image noise and one kind of occlusion only.
- The CS-vs-plain advantage at σ=0.03 (+4.7) is not significant with 5
  seeds.
