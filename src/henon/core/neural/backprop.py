"""
Entraînement du réseau par rétropropagation.

Ce module implémente la classe BackPropagation qui gère
l'entraînement du MLP par descente de gradient.
Utilise des opérations vectorisées NumPy pour la performance.
"""

from __future__ import annotations

import logging

import numpy as np

from henon.core.neural.losses import Loss, MSELoss
from henon.core.neural.network import Network

logger = logging.getLogger(__name__)


class BackPropagation:
    """
    Entraîneur par rétropropagation pour le MLP.

    Parameters
    ----------
    network : Network
        Réseau de neurones à entraîner.
    loss : Loss
        Fonction de perte.
    learning_rate : float
        Taux d'apprentissage.
    """

    def __init__(
        self,
        network: Network,
        loss: Loss | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        self.network = network
        self.loss = loss or MSELoss()
        self.learning_rate = learning_rate

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Entraîne le réseau sur un epoch complet (batch unique).

        Parameters
        ----------
        X : np.ndarray
            Données d'entrée (n_samples, n_inputs).
        y : np.ndarray
            Cibles (n_samples, n_outputs).

        Returns
        -------
        float
            Perte moyenne sur l'epoch.
        """
        # Propagation avant vectorisée
        y_pred = self.network.forward(X)

        # Calcul de la perte
        total_loss = self.loss(y_pred, y)

        # Rétropropagation
        grad = self.loss.derivative(y_pred, y)
        self.network.backward(grad)

        # Mise à jour des poids
        self.network.update_weights(self.learning_rate)

        return total_loss

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 500,
        patience: int = 30,
        verbose: bool = True,
    ) -> dict:
        """
        Entraîne le réseau avec early stopping optionnel.

        Parameters
        ----------
        X_train : np.ndarray
            Données d'entraînement (n_samples, n_inputs).
        y_train : np.ndarray
            Cibles d'entraînement (n_samples, n_outputs).
        X_val : np.ndarray | None
            Données de validation.
        y_val : np.ndarray | None
            Cibles de validation.
        epochs : int
            Nombre maximum d'epochs.
        patience : int
            Nombre d'epochs sans amélioration avant arrêt.
        verbose : bool
            Affiche les logs.

        Returns
        -------
        dict
            Historique de l'entraînement.
        """
        history = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Entraînement
            train_loss = self.train_epoch(X_train, y_train)
            history["train_loss"].append(train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                y_pred_val = self.network.forward(X_val)
                val_loss = float(self.loss(y_pred_val, y_val))
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if verbose and (epoch + 1) % 100 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} — "
                        f"Train: {train_loss:.6f} — Val: {val_loss:.6f}"
                    )

                if epochs_without_improvement >= patience:
                    if verbose:
                        logger.info(
                            f"Early stopping à l'epoch {epoch + 1}"
                        )
                    break
            else:
                if verbose and (epoch + 1) % 100 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} — Train: {train_loss:.6f}"
                    )

        return history
