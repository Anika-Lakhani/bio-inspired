# Iteration 13: Starting filter vs constraints (2×2 separation)

Date run: 2026-11-24 to 2026-11-25

## Purpose

The center-surround (CS) layer and the plain filter differ in two respects:
the **starting filter** (DoG vs random) and the **constraints** (kept
zero-sum with α ≥ 0, vs none). This run tests each separately by adding the
two missing corners of a 2×2 design.

All four corners use internal noise 0.03, a layer learning rate of 1e-2 and
3 epochs, over seeds 3–7. That's the regime where CS beat plain by +8.4 at
σ=0.01 in iteration 12, whose two corners and Control are reused here. The
two new corners ran in a rebuilt environment with the same package versions.

| | No constraints | Zero-sum + α ≥ 0 |
|---|---|---|
| Random start | Plain (iteration 12) | **Random start, constrained** (new) |
| DoG start | **DoG start, free** (new) | CS (iteration 12) |

## Results (accuracy %, mean ± sd over 5 seeds; internal noise ON at test)

| Model | Clean | σ=0.01 | σ=0.03 | Occluded |
|---|---|---|---|---|
| Control (no noise) | 91.0 ± 0.1 | 43.3 ± 2.2 | 9.4 | 89.0 |
| CS: DoG start + constraints | 90.3 ± 0.2 | 74.1 ± 7.2 | 16.9 ± 7.5 | 87.5 ± 0.5 |
| Random start + constraints | 90.1 ± 0.2 | 72.1 ± 7.7 | 15.7 ± 5.6 | 87.1 ± 0.8 |
| DoG start, no constraints | 89.2 ± 1.2 | 64.8 ± 15.8 | 17.3 ± 11.5 | 85.4 ± 2.9 |
| Plain: random start, no constraints | 90.0 ± 0.3 | 65.7 ± 11.2 | 12.1 ± 3.0 | 86.9 ± 1.2 |

**Effect of each factor at σ=0.01** (paired by seed):

| Change | Holding fixed | Difference | p | Seeds won |
|---|---|---|---|---|
| Add constraints | Random start | **+6.4** | 0.048 | 5/5 |
| Add constraints | DoG start | +9.3 | 0.21 | 3/5 |
| DoG start instead of random | No constraints | −0.9 | 0.87 | 3/5 |
| DoG start instead of random | Constraints | +2.1 | 0.37 | 4/5 |

**Learned strength α:**

| Model | α across seeds |
|---|---|
| Random start + constraints | +0.38 to +0.71 |
| CS (iteration 12) | +0.44 to +0.67 |
| DoG start, no constraints | ±1.0 to ±1.7, sign flips (2 negative, 3 positive) |
| Plain (iteration 12) | ±1.1 to ±1.5, sign flips |

## Findings

1. **The constraints, not the starting shape, account for the CS
   advantage.** A random filter with the same constraints performs like
   CS (72.1 vs 74.1, n.s.). A DoG-started filter without them performs
   like the plain filter (64.8 vs 65.7) and is the least stable model of
   all (±15.8 at σ=0.01).
2. **Constraints buy stability.** Both constrained models learn a moderate
   positive strength in every seed, with small spreads on clean and
   occluded images. Both unconstrained models grow α past ±1 with an
   unpredictable sign. Without the zero-sum rule, a DoG start does not stay
   center-surround; training pushes it elsewhere.
3. **The biological shape adds little beyond its constraint.** Starting
   from a DoG instead of a random filter changed nothing significant in
   either column.

## Interpretation

The zero-sum rule is the abstract principle behind lateral inhibition: a
unit's output is its own input *minus* a share of its neighbors', so
activity is redistributed, never added. That principle, not the specific
DoG shape, is what kept a fast-learning layer stable under internal noise.
The paper's "guard rail" finding is refined to: **the inhibitory
constraint is the guard rail; the center-surround shape is incidental.**

## Caveats

- Only 5 seeds. Only the random-start contrast at σ=0.01 is significant.
  The DoG-start contrast (+9.3) is large but noisy because the unconstrained
  DoG model varies so much.
- The constrained random filter wasn't inspected to see whether it learned
  a center-surround-like shape on its own.
- Two corners come from iteration 12, run before the environment was
  rebuilt with the same package versions.
