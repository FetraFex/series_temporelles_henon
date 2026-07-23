"""
Évaluation complète des dimensions d'embedding.

Ce module orchestre les étapes 1 à 6 de l'algorithme de Takens :
construction des vecteurs retardés, matrice de covariance,
décomposition spectrale, et calcul de l'erreur pour chaque dimension.
"""

from __future__ import annotations

import numpy as np

from .covariance import compute_covariance_matrix
from .dimension import build_delay_vectors
from .eigen import compute_eigendecomposition


def evaluate_embedding_dimensions(
    series: np.ndarray | list[float],
    delay: int = 1,
    min_dimension: int = 2,
    max_dimension: int = 20,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Évalue l'erreur d'embedding pour toutes les dimensions candidates.

    Procédure (calcul spectral UNIQUE — étape 6 du cours) :
      1. build_delay_vectors() appelée UNE SEULE FOIS, dimension = max_dimension
      2. compute_covariance_matrix() appelée UNE SEULE FOIS → Θ
      3. compute_eigendecomposition() appelée UNE SEULE FOIS → toutes les λⱼ
      4. Pour m de min_dimension à max_dimension : E(m) calculé vectoriellement

    Parameters
    ----------
    series : np.ndarray | list[float]
        Série temporelle scalaire.
    delay : int
        Délai de reconstruction r.
    min_dimension : int
        Dimension minimale à tester.
    max_dimension : int
        Dimension maximale à tester (détermine aussi la taille
        de la matrice de covariance — calcul spectral unique).
    center : bool
        Si True, centre la matrice de covariance.

    Returns
    -------
    dimensions : np.ndarray
        Tableau [min_dimension, ..., max_dimension].
    errors : np.ndarray
        Tableau des erreurs E(m) correspondantes.

    Raises
    ------
    ValueError
        Si max_dimension est trop grand pour la longueur de la série.
    """
    series = np.asarray(series, dtype=float)

    # Étape 2 : construction des vecteurs retardés (UNE SEULE FOIS)
    delay_vectors = build_delay_vectors(series, delay, max_dimension)

    # Étape 3 : matrice de covariance (UNE SEULE FOIS)
    cov_matrix = compute_covariance_matrix(delay_vectors, center=center)

    # Étape 4 : décomposition spectrale (UNE SEULE FOIS)
    eigenvalues, _ = compute_eigendecomposition(cov_matrix)

    # Étape 5 : calcul vectorisé de l'erreur pour chaque m
    # E(m) = √(λ_{m+1}) — erreur basée sur la première composante écartée
    dimensions = np.arange(min_dimension, max_dimension + 1)
    errors = np.zeros(len(dimensions), dtype=float)

    for idx, m in enumerate(dimensions):
        if m < len(eigenvalues):
            errors[idx] = np.sqrt(eigenvalues[m])
        else:
            errors[idx] = 0.0

    return dimensions, errors
