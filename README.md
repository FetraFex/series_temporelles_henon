# henon-forecasting

Forecasting the Hénon chaotic system using **Takens delay embedding** and a
**Multilayer Perceptron implemented entirely from scratch** (no TensorFlow,
PyTorch, Keras, or scikit-learn neural networks).

## Pipeline

1. Generate 500 values of the Hénon system (Xn, Yn)
2. Preprocess (validate, normalize, split)
3. Takens embedding → supervised dataset
4. From-scratch MLP (neurons, layers, forward, backprop, updates)
5. Training (LR, epochs, early stopping, checkpointing)
6. Multi-step prediction (1 / 3 / 10 / 20 steps)
7. Evaluation (MSE, RMSE, MAE, plots)
8. Visualization (attractor, phase space, learning & prediction curves)

## Structure

See `docs/architecture.md` for the full design rationale (clean architecture,
separation of concerns).

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_generation.py
python scripts/run_preprocessing.py
python scripts/run_embedding.py
python scripts/run_training.py
python scripts/run_prediction.py
python scripts/run_evaluation.py
```
