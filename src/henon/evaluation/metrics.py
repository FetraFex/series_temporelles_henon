"""
Évaluation du réseau de neurones.

Ce module fournit la classe Evaluation qui calcule les métriques
de performance : MSE, RMSE, MAE et NMSE.
"""

from __future__ import annotations

import numpy as np


class Evaluation:
    """
    Classe d'évaluation du réseau de neurones.

    Fournit des métriques de régression standards pour évaluer
    la qualité des prédictions du MLP.
    """

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error (Erreur quadratique moyenne).

        MSE = (1/n) * Σ(y_true - y_pred)²

        Parameters
        ----------
        y_true : np.ndarray
            Valeurs réelles.
        y_pred : np.ndarray
            Valeurs prédites.

        Returns
        -------
        float
            Valeur du MSE.
        """
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error (Racine de l'erreur quadratique moyenne).

        RMSE = √(MSE)

        Parameters
        ----------
        y_true : np.ndarray
            Valeurs réelles.
        y_pred : np.ndarray
            Valeurs prédites.

        Returns
        -------
        float
            Valeur du RMSE.
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error (Erreur absolue moyenne).

        MAE = (1/n) * Σ|y_true - y_pred|

        Parameters
        ----------
        y_true : np.ndarray
            Valeurs réelles.
        y_pred : np.ndarray
            Valeurs prédites.

        Returns
        -------
        float
            Valeur du MAE.
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Normalized Mean Squared Error (Erreur quadratique moyenne normalisée).

        NMSE = MSE / var(y_true)

        Le NMSE est sans unité et permet de comparer des séries
        de magnitudes différentes. Un NMSE < 1 indique que le modèle
        est meilleur qu'une prédiction constante (moyenne).

        Parameters
        ----------
        y_true : np.ndarray
            Valeurs réelles.
        y_pred : np.ndarray
            Valeurs prédites.

        Returns
        -------
        float
            Valeur du NMSE.
        """
        variance = np.var(y_true)
        if variance == 0:
            return 0.0
        return float(np.mean((y_true - y_pred) ** 2) / variance)

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calcule toutes les métriques en une seule fois.

        Parameters
        ----------
        y_true : np.ndarray
            Valeurs réelles.
        y_pred : np.ndarray
            Valeurs prédites.

        Returns
        -------
        dict
            Dictionnaire contenant MSE, RMSE, MAE, NMSE.
        """
        return {
            "mse": self.mse(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "nmse": self.nmse(y_true, y_pred),
        }
