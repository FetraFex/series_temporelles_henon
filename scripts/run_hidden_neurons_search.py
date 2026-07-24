"""
Script principal de recherche du nombre optimal de neurones cachés.

Ce script exécute l'expérimentation complète :
1. Chargement de la série de Hénon
2. Construction du dataset supervisé
3. Normalisation et split
4. Évaluation des architectures 6-H-1
5. Export des résultats et graphiques
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from henon.data.supervised import Dataset
from henon.evaluation.hidden_neurons_experiment import HiddenNeuronExperiment

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
XLSX_PATH = Path("data/raw/henon_500.xlsx")
EMBEDDING_DIM = 6
DELAY = 1
OUTPUT_DIR = Path("results/architecture_search")
HIDDEN_RANGE = [6, 7, 8, 9, 10]
N_REPETITIONS = 10
LEARNING_RATE = 0.01
EPOCHS = 500
PATIENCE = 30


def main() -> None:
    """Exécute la recherche du nombre optimal de neurones cachés."""
    logger.info("=" * 55)
    logger.info(" RECHERCHE DU NOMBRE OPTIMAL DE NEURONES CACHÉS")
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

    # --- Étape 4 : Expérimentation ---
    experiment = HiddenNeuronExperiment(
        dataset=dataset,
        hidden_range=HIDDEN_RANGE,
        n_repetitions=N_REPETITIONS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        patience=PATIENCE,
    )

    results_df = experiment.run()

    # --- Étape 5 : Résultats ---
    experiment.print_best(results_df)

    # --- Étape 6 : Export ---
    experiment.export_results(results_df, OUTPUT_DIR)
    experiment.plot_results(results_df, OUTPUT_DIR)

    logger.info(f"Terminé. Résultats dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
