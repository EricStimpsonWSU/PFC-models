# Usage examples

Notebooks that demonstrate how to use a model: setting up a run, reading the
output, driving a parameter sweep. One directory per catalogue model, named for
the model slug.

```
examples/<model-slug>/<what-it-demonstrates>.ipynb
```

These are documentation that happens to execute. They may be slow, they may
need a GPU, and nothing in CI fails if one goes stale.

**This is the only place notebooks live.** Not under `src/pfc_models/` — that
is importable library code — and not under `tests/`, which is reserved for
automated `pytest` tests with no human reading the output.

Each notebook opens with a markdown cell naming the model it uses, the
`pfc-models` version it was written against, and what the reader should take
away.

Output — figures, frame dumps, movies — is not committed. The notebook plus a
seed is the reproduction path.
