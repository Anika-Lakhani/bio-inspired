# Revisit (2026): nature's priors vs learned filters

A controlled re-examination of this project's lateral-inhibition and attention
experiments, reframed as a comparison: what does a biological mechanism add,
compared with the closest ordinary layer a CNN could learn?

- `REFRAMING.md`: the framing, research questions and current thesis.
- `iteration-NN-*/`: one folder per experiment, each with `experiment.py`
  (the exact script run), `results.json` (per-seed accuracies), `log.txt`
  and `NOTES.md` (what was tested, results, findings and caveats).

Iteration 1 reruns the original design with its bugs fixed (noise clamp,
ablation flags, parameter count). Iterations 7–8 contain a result later shown
to be a learning-rate artifact; their notes carry the correction.

Experiments load `ControlCNN` and `ExperimentalCNN` from `internal-code.ipynb`
(cell 0) and expect CIFAR-10 in `./data`. Each is run as
`python experiment.py path/to/internal-code.ipynb [previous results.json ...]`.
