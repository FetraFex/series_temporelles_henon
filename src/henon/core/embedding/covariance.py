"""
Construction de la matrice de covariance.

Ce module calcule la matrice de covariance (ou de corrélation non centrée)
à partir des vecteurs retardés, selon la méthode du cours
(ACP / Karhunen-Loève — Broomhead & King).
"""

from __future__ import annotations

import numpy as np


def compute_covariance_matrix(
    delay_vectors: np.ndarray,
    center: bool = False,
) -> np.ndarray:
    """
    Calcule la matrice de covariance des vecteurs retardés.

    Formule (matrice non centrée — méthode du cours) :
        Θ = (1/M) * Σ(i=1 à M) x̄(i) · x̄(i)ᵀ

    où M est le nombre de vecteurs (delay_vectors.shape[0]).

    Parameters
    ----------
    delay_vectors : np.ndarray
        Matrice de shape (M, dimension) contenant les vecteurs retardés.
    center : bool
        Si True, soustrait la moyenne des vecteurs avant le produit
        extérieur (vraie covariance statistique centrée).
        Si False (défaut), utilise la matrice non centrée, conformément
        à la méthode du cours (Broomhead & King).

    Returns
    -------
    np.ndarray
        Matrice carrée de shape (dimension, dimension).

    Notes
    -----
    La matrice non centrée (center=False) estSymétrique semi-définie
    positive et possède des valeurs propres réelles non négatives.
    La matrice centrée (center=True) est la covariance statistique
    classique, utile pour d'autres méthodes mais pas celle du cours.
    """
    delay_vectors = np.asarray(delay_vectors, dtype=float)
    m_vectors, dimension = delay_vectors.shape

    if center:
        # Covariance statistique classique (centrée)
        mean = delay_vectors.mean(axis=0)
        centered = delay_vectors - mean
        cov = (centered.T @ centered) / m_vectors
    else:
        # Méthode du cours : matrice non centrée
        cov = (delay_vectors.T @ delay_vectors) / m_vectors

    return cov
