# Reframing the paper: what nature's priors do, compared with learned CNN filters

## Old framing (drop this)

> "Biologically inspired mechanisms will improve CNN performance."

This is a one-sided prediction. When it doesn't hold (iterations 1, 5 and
6), the result reads as a failure, even when it's informative.

## New framing

> **What does nature's solution buy a network, compared with what a
> standard CNN learns for itself?**

Treat the biological mechanisms as one design approach (fixed priors shaped
by evolution) and standard CNN training as another (filters learned from
data). Compare them head to head under controlled conditions. No outcome
counts as failure:

- If the biological version wins, the prior encodes something that training
  doesn't find.
- If they tie, training rediscovers what the prior encodes.
- If the traditional version wins, the prior is a constraint the task
  doesn't need.

## Suggested research questions

1. **Placement:** does the biological mechanism matter more where biology
   uses it (the earliest stage) than at a late stage? (Iterations 1–3)
2. **Form vs capacity:** with equal extra capacity at the same spot, does
   the center-surround form beat an unconstrained learned filter?
   (Iterations 4–6)
3. **Learned vs fixed:** does imposing the prior (as evolution does) differ
   from letting training shape it? (Iteration 6)
4. **Priors and noise together:** biological vision runs on noisy neurons.
   Does the prior become more valuable when the network has to cope with
   internal noise? (Iterations 7–8)

## What the evidence says so far

| Question | Answer | Iterations |
|---|---|---|
| Naively bolting on the mechanisms | Hurts (bugs plus bad design) | 1 |
| Placement | Early helps, late doesn't, matching biology | 2–3 |
| Form vs capacity, no internal noise | Tie: CNNs learn an equally good filter | 4–5 |
| Fixed vs learned | Fixed is more consistent but costs on occlusion | 6 |
| Fixed prior vs learned filter with internal noise | Tie once the learning rate is matched or training is longer (the earlier "prior wins" result was a confound) | 7–10 |
| Learnable center-surround + internal noise | Largest input-noise gain in the project (+10 at σ=0.03, +8 at σ=0.05) at a fast learning rate, where it stays stable and the plain filter doesn't (+9 clean, +10.5 occluded vs plain). At a moderate rate the advantage shrinks to a small edge under mild noise | 9, 11 |
| Longer training (3 epochs, 3 learning rates) | Training with internal noise costs about 1 point on clean images and gains +14 to +31 at σ=0.01; the CS advantage over the plain filter replicates at the fast rate only (+8.4, 5/5 seeds); networks come to depend on the noise | 10, 12 |

## Suggested thesis

> "When biological mechanisms were compared with ordinary learned layers
> under matched conditions, the biological form itself was not a better
> filter. Training with neuron-like internal noise mattered far more: with
> enough training it cost about 1 point of clean accuracy and recovered up
> to 31 points under mild image noise, and networks came to depend on the
> noise as part of their computation, as biological circuits do. What the
> center-surround form added was stability. It let an early layer learn
> aggressively under noise without becoming erratic, and in that regime it
> gave the most robust model in the study (+8 points over an unconstrained
> layer, replicated at 1 and 3 epochs). An earlier apparent advantage for a
> fixed biological prior turned out to be a learning-rate artifact, which
> shows how easily such comparisons mislead without matched controls."

The iteration 1 results stay in the paper as the starting point: the
original design and its bugs, and why the redesign was needed.
