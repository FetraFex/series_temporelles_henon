"""
Script de prédiction à un pas du réseau neuronal entraîné.

Ce script entraîne le réseau MLP 6-8-1 (Sigmoid/Linear) sur la série
de Hénon, puis effectue une prédiction pour chaque prototype du jeu
de test et sauvegarde les résultats.

Paramètres d'entraînement :
- Architecture : 6-8-1
- Taux d'apprentissage : 0.1
- Maximum d'époques : 1000
- Tolérance : 1e-5
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from henon.data.supervised import Dataset
from henon.core.neural.network import Network
from henon.core.neural.losses import MSELoss
from henon.core.neural.backprop import BackPropagation

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
TOLERANCE = 1e-5


def main() -> None:
    """Exécute la prédiction à un pas sur le jeu de test."""
    logger.info("=" * 55)
    logger.info(" PRÉDICTION À UN PAS — RÉSEAU 6-8-1")
    logger.info("=" * 55)

    # --- Étape 1 : Chargement du dataset ---
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

    # Construction des données supervisées
    dataset.build_supervised()
    logger.info(f"Données supervisées : X={dataset.X.shape}, y={dataset.y.shape}")

    # Normalisation et split
    dataset.normalize(method="minmax")
    dataset.split()

    # --- Étape 2 : Création et entraînement du réseau ---
    network = Network(
        layer_sizes=[6, 8, 1],
        activations=["sigmoid", "linear"],
        seed=SEED,
    )
    logger.info(f"Réseau créé : {network.layer_sizes}")

    # Entraînement du réseau
    logger.info("Entraînement du réseau...")
    loss_fn = MSELoss()
    trainer = BackPropagation(
        network=network,
        loss=loss_fn,
        learning_rate=LEARNING_RATE,
    )

    start_time = time.perf_counter()
    prev_train_loss = float("inf")

    for epoch in range(MAX_EPOCHS):
        train_loss = trainer.train_epoch(dataset.X_train, dataset.y_train)

        if abs(train_loss - prev_train_loss) < TOLERANCE:
            logger.info(f"Convergence à l'époque {epoch + 1}")
            break
        prev_train_loss = train_loss

        if (epoch + 1) % 200 == 0:
            logger.info(f"Époque {epoch + 1}/{MAX_EPOCHS} — MSE: {train_loss:.6f}")

    elapsed = time.perf_counter() - start_time
    logger.info(f"Entraînement terminé en {elapsed:.4f}s")

    # --- Étape 3 : Prédiction sur le jeu de test ---
    logger.info("Prédiction sur le jeu de test...")

    # Propagation avant sur tous les prototypes de test
    # Aucun apprentissage n'est effectué
    y_pred_norm = network.predict(dataset.X_test)

    # Dénormalisation pour obtenir les valeurs réelles
    y_pred = dataset.denormalize_y(y_pred_norm).flatten()
    y_true = dataset.denormalize_y(dataset.y_test).flatten()

    # Calcul des erreurs
    errors = y_true - y_pred

    # --- Étape 4 : Sauvegarde des résultats ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "one_step_predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_value", "predicted_value", "error"])
        for i in range(len(y_true)):
            writer.writerow([
                i,
                f"{y_true[i]:.10f}",
                f"{y_pred[i]:.10f}",
                f"{errors[i]:.10f}",
            ])

    logger.info(f"Résultats sauvegardés : {csv_path}")

    # --- Étape 5 : Affichage des résultats ---
    print("")
    print("=" * 40)
    print("  ONE STEP PREDICTION")
    print("=" * 40)
    print("")
    print(f"  Nombre de prédictions : {len(y_true)}")
    print("")

    # Affichage de quelques exemples
    print("  Exemples :")
    print("")

    for i in range(min(5, len(y_true))):
        print(f"  Prototype {i} :")
        print(f"    Réel   : {y_true[i]:.8f}")
        print(f"    Prédit : {y_pred[i]:.8f}")
        print(f"    Erreur : {errors[i]:.8f}")
        print("")

    # Calcul des métriques globales
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    print("  Métriques globales :")
    print("")
    print(f"    MSE  : {mse:.8f}")
    print(f"    RMSE : {rmse:.8f}")
    print(f"    MAE  : {mae:.8f}")
    print("")

    # Affichage d'un exemple détaillé
    print("=" * 40)
    print("  EXEMPLE DÉTAILLÉ (prototype 0)")
    print("=" * 40)
    print("")
    print(f"  Réel :")
    print(f"    {y_true[0]:.8f}")
    print("")
    print(f"  Prédit :")
    print(f"    {y_pred[0]:.8f}")
    print("")
    print(f"  Erreur :")
    print(f"    {errors[0]:.8f}")
    print("")

    # Commentaire pédagogique
    print("=" * 40)
    print("  COMMENTAIRE PÉDAGOGIQUE")
    print("=" * 40)
    print("")
    print("  Cette prédiction utilise uniquement la propagation avant")
    print("  (forward pass) du réseau entraîné. Aucun apprentissage")
    print("  n'est effectué — les poids sont fixes.")
    print("")
    print("  Pour chaque prototype :")
    print("    1. Les 6 entrées (vecteur de Takens) traversent le réseau")
    print("    2. La couche cachée applique : h_j = sigmoid(z_j)")
    print("    3. La couche de sortie applique : y = Linear(z)")
    print("    4. La prédiction est comparée à la valeur réelle")
    print("")
    print("  L'erreur mesure l'écart entre la prédiction du modèle")
    print("  et la valeur observée de la série de Hénon.")
    print("=" * 40)
    print("")


if __name__ == "__main__":
    main()
