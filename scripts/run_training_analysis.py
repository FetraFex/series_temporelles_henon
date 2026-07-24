"""
Script d'analyse et de visualisation de l'apprentissage du réseau.

Ce script entraîne le réseau 6-8-1 sur la série de Hénon, enregistre
l'historique d'apprentissage à chaque époque, puis génère les graphiques
et les fichiers de synthèse.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from henon.data.supervised import Dataset
from henon.core.neural.network import Network
from henon.core.neural.losses import MSELoss
from henon.core.neural.backprop import BackPropagation
from analysis.training_history import TrainingHistory

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
XLSX_PATH = Path("data/raw/henon_500.xlsx")
EMBEDDING_DIM = 6
DELAY = 1
OUTPUT_DIR = Path("outputs")
SEED = 42
LEARNING_RATE = 0.1
MAX_EPOCHS = 1000
PATIENCE = 30
TOLERANCE = 1e-5


def main() -> None:
    """Exécute l'entraînement et l'analyse de l'historique."""
    logger.info("=" * 55)
    logger.info(" ANALYSE DE L'APPRENTISSAGE DU RESEAU 6-8-1")
    logger.info("=" * 55)

    # --- Étape 1 : Chargement des données ---
    if not XLSX_PATH.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé : {XLSX_PATH}. "
            f"Exécutez d'abord scripts/run_generation.py."
        )

    dataset = Dataset.from_excel(
        path=XLSX_PATH,
        column="Xn",
        embedding_dim=EMBEDDING_DIM,
        delay=DELAY,
    )
    logger.info(f"Série chargée : {len(dataset.series)} points")

    # --- Étape 2 : Construction du dataset supervisé ---
    dataset.build_supervised()
    logger.info(f"Données supervisées : X={dataset.X.shape}, y={dataset.y.shape}")

    # --- Étape 3 : Normalisation et split ---
    dataset.normalize(method="minmax")
    dataset.split()

    # --- Étape 4 : Création du réseau ---
    network = Network(
        layer_sizes=[6, 8, 1],
        activations=["sigmoid", "linear"],
        seed=SEED,
    )
    logger.info(f"Réseau créé : {network.layer_sizes}")

    # --- Étape 5 : Entraînement avec suivi ---
    history = TrainingHistory(output_dir=OUTPUT_DIR)
    history.set_metadata(
        architecture="6-8-1",
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        tolerance=TOLERANCE,
    )

    loss_fn = MSELoss()
    trainer = BackPropagation(
        network=network,
        loss=loss_fn,
        learning_rate=LEARNING_RATE,
    )

    logger.info("Début de l'entraînement...")
    start_time = time.perf_counter()

    # Entraînement épique par épique pour le suivi
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    prev_train_loss = float("inf")

    for epoch in range(MAX_EPOCHS):
        # Entraînement sur une époque
        train_loss = trainer.train_epoch(
            dataset.X_train, dataset.y_train
        )

        # Calcul de la validation
        y_pred_val = network.forward(dataset.X_val)
        val_loss = float(loss_fn(y_pred_val, dataset.y_val))

        # Enregistrement
        history.record(epoch + 1, train_loss, val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Tolérance (convergence)
        if abs(train_loss - prev_train_loss) < TOLERANCE:
            logger.info(
                f"Convergence à l'époque {epoch + 1} "
                f"(variation < {TOLERANCE})"
            )
            break
        prev_train_loss = train_loss

        if epochs_without_improvement >= PATIENCE:
            logger.info(f"Early stopping à l'époque {epoch + 1}")
            break

        if (epoch + 1) % 100 == 0:
            logger.info(
                f"Époque {epoch + 1}/{MAX_EPOCHS} — "
                f"Train: {train_loss:.6f} — Val: {val_loss:.6f}"
            )

    elapsed = time.perf_counter() - start_time
    history.training_time = elapsed
    history.best_val_loss = best_val_loss

    logger.info(f"Entraînement terminé en {elapsed:.4f}s")

    # --- Étape 6 : Export des résultats ---
    logger.info("Export des résultats...")
    paths = history.export_all()

    for p in paths:
        logger.info(f"  → {p}")

    # --- Étape 7 : Affichage du résumé ---
    print("\n" + "=" * 40)
    print("  TRAINING SUMMARY")
    print("=" * 40)
    print(f"  Architecture : {history.architecture}")
    print(f"  Initial loss : {history.initial_loss:.10f}")
    print(f"  Final loss   : {history.final_loss:.10f}")
    print(f"  Best val loss: {history.best_val_loss:.10f}")
    print(f"  Best epoch   : {history.best_epoch}")
    print(f"  Total epochs : {history.total_epochs}")
    print(f"  Training time: {history.training_time:.4f}s")
    print("=" * 40 + "\n")

    logger.info(f"Terminé. Résultats dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
