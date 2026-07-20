# Usage

Each pipeline stage is run via a script in `scripts/`. Configuration is read
from the matching file in `configs/`.

```bash
python scripts/run_generation.py      # Stage 1
python scripts/run_preprocessing.py   # Stage 2
python scripts/run_embedding.py       # Stage 3
python scripts/run_training.py        # Stage 5
python scripts/run_prediction.py      # Stage 6
python scripts/run_evaluation.py      # Stage 7
```

Outputs:
- Datasets → `data/`
- Models → `models/`
- Figures & metrics → `results/`
