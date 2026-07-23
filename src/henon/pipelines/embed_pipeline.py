"""
Pipeline d'embedding de Takens.

Ce module orchestre les étapes complètes de l'algorithme :
lecture → vecteurs retardés → covariance → valeurs propres →
erreur → plateau → visualisation → sauvegarde.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from henon.core.embedding.covariance import compute_covariance_matrix
from henon.core.embedding.dimension import build_delay_vectors
from henon.core.embedding.eigen import compute_eigendecomposition
from henon.core.embedding.error import compute_embedding_error
from henon.core.embedding.evaluation import evaluate_embedding_dimensions
from henon.core.embedding.plateau import find_first_plateau
from henon.utils.config import load_config
from henon.visualization.embedding_error import plot_embedding_error

logger = logging.getLogger(__name__)


def run_embedding_pipeline(
    series: np.ndarray | list[float] | None = None,
    config_name: str = "takens",
) -> dict:
    """
    Exécute le pipeline complet de Takens.

    Parameters
    ----------
    series : np.ndarray | list[float] | None
        Série temporelle à analyser. Si None, charge depuis data/raw/.
    config_name : str
        Nom du fichier de configuration YAML.

    Returns
    -------
    dict
        Résultats du pipeline : dimensions, errors, optimal_dimension,
        eigenvalues, cov_matrix.
    """
    # --- Lecture de la configuration ---
    config = load_config(config_name)
    delay = config["delay"]
    min_dim = config["min_dimension"]
    max_dim = config["max_dimension"]
    threshold = config["plateau_threshold"]
    min_consec = config["plateau_min_consecutive"]
    center = config["center_covariance"]
    output_dir = Path(config["output_directory"])

    logger.info(f"Configuration Takens chargée : r={delay}, m=[{min_dim}..{max_dim}]")

    # --- Étape 1 : Lecture de la série ---
    if series is None:
        xlsx_path = Path("data/raw/henon_500.xlsx")
        if not xlsx_path.exists():
            raise FileNotFoundError(
                f"Fichier non trouvé : {xlsx_path}. "
                f"Exécutez d'abord scripts/run_generation.py."
            )
        df = pd.read_excel(xlsx_path)
        series = df["Xn"].values
        logger.info(f"Série chargée depuis {xlsx_path} ({len(series)} points)")
    else:
        series = np.asarray(series, dtype=float)
        logger.info(f"Série fournie directement ({len(series)} points)")

    # --- Étapes 2-5 : Évaluation vectorisée (calcul spectral UNIQUE) ---
    dimensions, errors = evaluate_embedding_dimensions(
        series=series,
        delay=delay,
        min_dimension=min_dim,
        max_dimension=max_dim,
        center=center,
    )

    # --- Étape 6 : Calcul complet pour la matrice et les valeurs propres ---
    delay_vectors = build_delay_vectors(series, delay, max_dim)
    cov_matrix = compute_covariance_matrix(delay_vectors, center=center)
    eigenvalues, eigenvectors = compute_eigendecomposition(cov_matrix)

    logger.info(f"Valeurs propres (top 5) : {eigenvalues[:5]}")

    # --- Étape 7 : Détection du plateau ---
    optimal_dim = find_first_plateau(
        dimensions, errors,
        threshold=threshold,
        min_consecutive=min_consec,
    )
    logger.info(f"Dimension optimale détectée : {optimal_dim}")

    # --- Étape 8 : Visualisation ---
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_embedding_error(
        dimensions=dimensions.tolist(),
        errors=errors.tolist(),
        optimal_dimension=optimal_dim,
        output_dir=str(output_dir),
        show=False,
    )

    # --- Étape 9 : Sauvegarde ---
    # Matrice de covariance
    pd.DataFrame(cov_matrix).to_csv(output_dir / "covariance_matrix.csv", index=False)

    # Valeurs propres
    pd.DataFrame({
        "eigenvalue": eigenvalues,
        "index": range(len(eigenvalues)),
    }).to_csv(output_dir / "eigenvalues.csv", index=False)

    # Erreurs d'embedding
    pd.DataFrame({
        "dimension": dimensions,
        "error": errors,
    }).to_csv(output_dir / "embedding_errors.csv", index=False)

    # Dimension optimale
    with open(output_dir / "embedding_dimension.txt", "w", encoding="utf-8") as f:
        f.write(f"Dimension optimale : {optimal_dim}\n")
        f.write(f"Methode : ACP / Karhunen-Loève (Broomhead & King)\n")
        f.write(f"Seuil : {threshold}, Pas consécutifs : {min_consec}\n")
        f.write(f"Plage testée : [{min_dim}, {max_dim}]\n")

    logger.info(f"Résultats sauvegardés dans {output_dir}/")

    return {
        "dimensions": dimensions,
        "errors": errors,
        "optimal_dimension": optimal_dim,
        "eigenvalues": eigenvalues,
        "cov_matrix": cov_matrix,
    }
