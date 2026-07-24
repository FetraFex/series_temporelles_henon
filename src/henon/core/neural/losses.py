"""
Fonctions de perte pour l'entraînement du réseau.

Ce module implémente la fonction de perte MSE (Mean Squared Error).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """Interface abstraite pour les fonctions de perte."""

    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calcule la perte moyenne."""

    @abstractmethod
    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Gradient de la perte par rapport à y_pred."""


class MSELoss(Loss):
    """
    MSE = (1/n) * Σ(y_pred - y_true)²
    Gradient : 2/n * (y_pred - y_true)
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / n


LOSSES = {"mse": MSELoss}


def get_loss(name: str) -> Loss:
    if name not in LOSSES:
        raise ValueError(f"Perte inconnue : '{name}'.")
    return LOSSES[name]()
