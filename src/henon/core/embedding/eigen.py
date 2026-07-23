"""
Décomposition en valeurs propres.

Ce module calcule la décomposition spectrale de la matrice
de covariance symétrique via numpy.linalg.eigh().
"""

from __future__ import annotations

import numpy as np


def compute_eigendecomposition(
    covariance_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Décompose la matrice de covariance en valeurs et vecteurs propres.

    Utilise numpy.linalg.eigh() (réservé aux matrices symétriques),
    qui retourne des valeurs propres réelles et des vecteurs propres
    orthogonaux.

    Parameters
    ----------
    covariance_matrix : np.ndarray
        Matrice carrée symétrique (dimension × dimension).

    Returns
    -------
    eigenvalues : np.ndarray
        Valeurs propres triées par ordre DÉCROISSANT.
    eigenvectors : np.ndarray
        Vecteurs propres correspondants, dans le même ordre.

    Notes
    -----
    Les valeurs propres légèrement négatives (bruit numérique sur
    une matrice quasi-singulière) sont clipées à 0, car la matrice
    de covariance non centrée est mathématiquement semi-définie
    positive.
    """
    # eigh retourne les valeurs propres en ordre croissant
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Tri décroissant
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Clip des valeurs propres négatives (bruit numérique)
    # La matrice de covariance non centrée est semi-définie positive,
    # donc toute valeur propre négative est un artefact d'arrondi.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    return eigenvalues, eigenvectors
