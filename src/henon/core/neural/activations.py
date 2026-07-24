"""
Fonctions d'activation pour le réseau de neurones.

Ce module implémente les fonctions d'activation utilisées
dans le MLP : tanh, linéaire et sigmoïde.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Interface abstraite pour les fonctions d'activation."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Calcule la sortie de la fonction d'activation."""

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcule la dérivée de la fonction d'activation."""


class Tanh(Activation):
    """tanh(x) ∈ [-1, 1], dérivée : 1 - tanh²(x)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2


class Linear(Activation):
    """Identité : f(x) = x, dérivée : 1."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class Sigmoid(Activation):
    """σ(x) = 1/(1+exp(-x)), dérivée : σ(x)*(1-σ(x))."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.__call__(x)
        return s * (1.0 - s)


ACTIVATIONS = {"tanh": Tanh, "linear": Linear, "sigmoid": Sigmoid}


def get_activation(name: str) -> Activation:
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Activation inconnue : '{name}'. "
            f"Choix possibles : {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]()
