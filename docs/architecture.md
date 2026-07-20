# Architecture

This project follows a **Clean Architecture**-inspired layout designed for a
research Master's thesis.

## Layers (dependency rule: outer depends on inner)

```
pipelines/  ->  training/ evaluation/ visualization/  ->  data/  ->  core/  ->  utils/
```

- **`core/`** — pure science, no I/O, no plotting. The MLP under
  `core/neural/` is fully dependency-free (no TF/PyTorch/Keras/sklearn).
- **`data/`** — dataset lifecycle (generation, loading, validation, preprocessing).
- **`training/`**, **`evaluation/`** — experiment orchestration and metrics.
- **`visualization/`** — all matplotlib figures.
- **`utils/`** — cross-cutting helpers (config, constants, logging, io).
- **`pipelines/`** — high-level workflows tying stages together.

## Separation of concerns

- Configuration lives in `configs/*.yaml`, not in code.
- The from-scratch neural network is isolated so it can be tested and extended
  (new activations, optimizers, layers) without touching I/O or plotting.
- `scripts/` are thin CLI entry points; real logic is in `src/henon/`.

## Extension points

- New dynamical systems → `core/dynamical_systems/`
- New embeddings → `core/embedding/`
- New activations/optimizers → `core/neural/`
- New stages → add a pipeline + script + config
