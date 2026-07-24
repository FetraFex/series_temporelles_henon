"""
Script principal d'optimisation des paramètres d'entraînement.

Ce script exécute les trois expérimentations :
1. Taux d'apprentissage (Learning Rate)
2. Nombre maximum d'epochs
3. Tolérance d'erreur

Architecture fixe : 6-8-1 (Sigmoid cachée, Linéaire en sortie).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from henon.data.supervised import Dataset
from henon.evaluation.training_parameter_experiment import (
    TrainingParameterExperiment,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
XLSX_PATH = Path("data/raw/henon_500.xlsx")
EMBEDDING_DIM = 6
DELAY = 1
OUTPUT_DIR = Path("results/training_parameters")
N_REPETITIONS = 10

# Paramètres à tester
LEARNING_RATES = [0.001, 0.005, 0.01, 0.05, 0.1]
EPOCHS_LIST = [500, 1000, 2000, 5000]
TOLERANCES = [1e-2, 1e-3, 1e-4, 1e-5]

# Paramètres fixes par défaut
DEFAULT_LR = 0.01
DEFAULT_EPOCHS = 500
DEFAULT_PATIENCE = 30
DEFAULT_TOLERANCE = 1e-5


def main() -> None:
    """Exécute l'optimisation des paramètres d'entraînement."""
    logger.info("=" * 55)
    logger.info(" OPTIMISATION DES PARAMÈTRES D'ENTRAÎNEMENT")
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
    X, y = dataset.build_supervised()
    logger.info(f"Données supervisées : X={X.shape}, y={y.shape}")

    # --- Étape 3 : Normalisation et split ---
    dataset.normalize(method="minmax")
    dataset.split()

    # --- Étape 4 : Création de l'expérimentation ---
    experiment = TrainingParameterExperiment(
        dataset=dataset,
        n_repetitions=N_REPETITIONS,
    )

    # --- Étape 5 : Expérimentation 1 — Learning Rate ---
    logger.info("-" * 55)
    logger.info("  EXPÉRIMENTATION 1 : TAUX D'APPRENTISSAGE")
    logger.info("-" * 55)
    lr_results = experiment.run_learning_rate(
        learning_rates=LEARNING_RATES,
        epochs=DEFAULT_EPOCHS,
        patience=DEFAULT_PATIENCE,
        tolerance=DEFAULT_TOLERANCE,
    )

    best_lr = experiment.compare(lr_results, "learning_rate")
    logger.info(f"Meilleur LR : {best_lr['param_value']} "
                f"(MSE = {best_lr['avg_mse']:.8f})")

    # --- Étape 6 : Expérimentation 2 — Epochs ---
    logger.info("-" * 55)
    logger.info("  EXPÉRIMENTATION 2 : NOMBRE MAX D'EPOCHS")
    logger.info("-" * 55)
    epochs_results = experiment.run_epochs(
        epochs_list=EPOCHS_LIST,
        learning_rate=best_lr["param_value"],
        patience=DEFAULT_PATIENCE,
        tolerance=DEFAULT_TOLERANCE,
    )

    best_epochs = experiment.compare(epochs_results, "max_epochs")
    logger.info(f"Meilleur max_epochs : {int(best_epochs['param_value'])} "
                f"(MSE = {best_epochs['avg_mse']:.8f})")

    # --- Étape 7 : Expérimentation 3 — Tolérance ---
    logger.info("-" * 55)
    logger.info("  EXPÉRIMENTATION 3 : TOLÉRANCE D'ERREUR")
    logger.info("-" * 55)
    tol_results = experiment.run_tolerance(
        tolerances=TOLERANCES,
        learning_rate=best_lr["param_value"],
        epochs=DEFAULT_EPOCHS,
        patience=DEFAULT_PATIENCE,
    )

    best_tol = experiment.compare(tol_results, "tolerance")
    logger.info(f"Meilleure tolerance : {best_tol['param_value']} "
                f"(MSE = {best_tol['avg_mse']:.8f})")

    # --- Étape 8 : Rapport final ---
    experiment.print_best(lr_results, epochs_results, tol_results)

    # --- Étape 9 : Export ---
    experiment.export_results(lr_results, epochs_results, tol_results, OUTPUT_DIR)
    experiment.plot_results(lr_results, epochs_results, tol_results, OUTPUT_DIR)

    logger.info(f"Terminé. Résultats dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
