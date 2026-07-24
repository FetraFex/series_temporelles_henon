"""
Module de prédiction pour le réseau de neurones.

Ce module fournit une interface simple pour effectuer des prédictions
avec un réseau déjà entraîné.
"""

from __future__ import annotations

import numpy as np

from henon.core.neural.network import Network


class Prediction:
    """
    Interface de prédiction pour un réseau entraîné.

    Parameters
    ----------
    network : Network
        Réseau de neurones entraîné.
    """

    def __init__(self, network: Network) -> None:
        self.network = network

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la sortie pour un ou plusieurs échantillons.

        Parameters
        ----------
        X : np.ndarray
            Données d'entrée (n_samples, n_inputs) ou (n_inputs,).

        Returns
        -------
        np.ndarray
            Prédictions (n_samples, n_outputs) ou (n_outputs,).
        """
        return self.network.predict(X)

    def predict_recursive(
        self,
        initial_conditions: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Prédiction récursive (autoregressive).

        Parameters
        ----------
        initial_conditions : np.ndarray
            Conditions initiales de forme (n_inputs,).
        n_steps : int
            Nombre de pas de temps à prédire.

        Returns
        -------
        np.ndarray
            Série prédite de forme (n_steps,).
        """
        predictions = []
        current = initial_conditions.copy()

        for _ in range(n_steps):
            output = self.network.forward(current)
            predictions.append(output[0])
            current = np.roll(current, -1)
            current[-1] = output[0]

        return np.array(predictions)
