"""
Dataset supervisé pour l'apprentissage du MLP.

Ce module fournit la classe Dataset qui transforme une série temporelle
en données supervisées (X, y) pour l'entraînement du réseau.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Dataset:
    """
    Dataset supervisé construit à partir d'une série temporelle.

    Parameters
    ----------
    series : np.ndarray
        Série temporelle scalaire.
    embedding_dim : int
        Dimension d'embedding (nombre d'entrées du réseau).
    delay : int
        Délai entre les composantes du vecteur retardé.
    train_ratio : float
        Proportion des données d'entraînement.
    val_ratio : float
        Proportion des données de validation.
    test_ratio : float
        Proportion des données de test.
    """

    def __init__(
        self,
        series: np.ndarray,
        embedding_dim: int = 6,
        delay: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Les ratios doivent sommer à 1.0, obtenu : "
                f"{train_ratio + val_ratio + test_ratio}"
            )

        self.series = np.asarray(series, dtype=float)
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Données brutes
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

        # Données normalisées
        self.X_norm: np.ndarray | None = None
        self.y_norm: np.ndarray | None = None
        self.norm_params: dict | None = None

        # Données splittées
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

    def build_supervised(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construit les paires (X, y) supervisées.

        Pour chaque instant t, on crée :
        - X[i] = [x[t], x[t-1], ..., x[t-(d-1)]]  (vecteur retardé)
        - y[i] = x[t+1]  (valeur suivante)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X de forme (N, embedding_dim), y de forme (N, 1).
        """
        N = len(self.series)
        M = N - (self.embedding_dim - 1) * self.delay - 1

        if M <= 0:
            raise ValueError(
                f"Série trop courte ({N} points) pour embedding_dim={self.embedding_dim} "
                f"et delay={self.delay}. Il faut au moins "
                f"{(self.embedding_dim - 1) * self.delay + 2} points."
            )

        # Construction vectorisée des vecteurs retardés
        offsets = np.arange(self.embedding_dim) * self.delay
        indices = np.arange(M)[:, None] + offsets
        self.X = self.series[indices]

        # Cibles : la valeur suivante
        target_indices = np.arange(self.embedding_dim * self.delay, M + self.embedding_dim * self.delay)
        self.y = self.series[target_indices].reshape(-1, 1)

        logger.info(
            f"Données supervisées : X={self.X.shape}, y={self.y.shape}"
        )
        return self.X, self.y

    def normalize(self, method: str = "minmax") -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Normalise les données X et y.

        Parameters
        ----------
        method : str
            Méthode de normalisation ('minmax' ou 'standard').

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict]
            X normalisé, y normalisé, paramètres de normalisation.
        """
        if self.X is None or self.y is None:
            raise ValueError("Construisez d'abord les données avec build_supervised().")

        if method == "minmax":
            # Normalisation min-max sur X
            x_min = self.X.min(axis=0)
            x_max = self.X.max(axis=0)
            x_range = x_max - x_min
            x_range[x_range == 0] = 1.0  # Éviter la division par zéro

            self.X_norm = (self.X - x_min) / x_range

            # Normalisation min-max sur y
            y_min = self.y.min()
            y_max = self.y.max()
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1.0

            self.y_norm = (self.y - y_min) / y_range

            self.norm_params = {
                "method": "minmax",
                "x_min": x_min,
                "x_max": x_max,
                "x_range": x_range,
                "y_min": y_min,
                "y_max": y_max,
                "y_range": y_range,
            }

        elif method == "standard":
            # Standardisation (z-score)
            x_mean = self.X.mean(axis=0)
            x_std = self.X.std(axis=0)
            x_std[x_std == 0] = 1.0

            self.X_norm = (self.X - x_mean) / x_std

            y_mean = self.y.mean()
            y_std = self.y.std()
            if y_std == 0:
                y_std = 1.0

            self.y_norm = (self.y - y_mean) / y_std

            self.norm_params = {
                "method": "standard",
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }
        else:
            raise ValueError(f"Méthode inconnue : '{method}'. Choix : minmax, standard")

        logger.info(f"Normalisation '{method}' appliquée")
        return self.X_norm, self.y_norm, self.norm_params

    def denormalize(self, X_norm: np.ndarray | None = None) -> np.ndarray:
        """
        Dénormalise les données X.

        Parameters
        ----------
        X_norm : np.ndarray | None
            Données à dénormaliser. Si None, dénormalise X_norm stocké.

        Returns
        -------
        np.ndarray
            Données dénormalisées.
        """
        if self.norm_params is None:
            raise ValueError("Normalisez d'abord les données avec normalize().")

        if X_norm is None:
            X_norm = self.X_norm

        params = self.norm_params
        if params["method"] == "minmax":
            return X_norm * params["x_range"] + params["x_min"]
        else:
            return X_norm * params["x_std"] + params["x_mean"]

    def denormalize_y(self, y_norm: np.ndarray | None = None) -> np.ndarray:
        """
        Dénormalise les données y.

        Parameters
        ----------
        y_norm : np.ndarray | None
            Données à dénormaliser. Si None, dénormalise y_norm stocké.

        Returns
        -------
        np.ndarray
            Données dénormalisées.
        """
        if self.norm_params is None:
            raise ValueError("Normalisez d'abord les données avec normalize().")

        if y_norm is None:
            y_norm = self.y_norm

        params = self.norm_params
        if params["method"] == "minmax":
            return y_norm * params["y_range"] + params["y_min"]
        else:
            return y_norm * params["y_std"] + params["y_mean"]

    def split(self) -> tuple:
        """
        Découpe les données en train/val/test (chronologique).

        Returns
        -------
        tuple
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if self.X_norm is None or self.y_norm is None:
            raise ValueError("Normalisez d'abord les données avec normalize().")

        n = len(self.X_norm)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        self.X_train = self.X_norm[:n_train]
        self.y_train = self.y_norm[:n_train]

        self.X_val = self.X_norm[n_train:n_train + n_val]
        self.y_val = self.y_norm[n_train:n_train + n_val]

        self.X_test = self.X_norm[n_train + n_val:]
        self.y_test = self.y_norm[n_train + n_val:]

        logger.info(
            f"Split : train={len(self.X_train)}, "
            f"val={len(self.X_val)}, test={len(self.X_test)}"
        )

        return (
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
        )

    @classmethod
    def from_excel(
        cls,
        path: str | Path,
        column: str = "Xn",
        embedding_dim: int = 6,
        delay: int = 1,
        **kwargs,
    ) -> Dataset:
        """
        Charge les données depuis un fichier Excel.

        Parameters
        ----------
        path : str | Path
            Chemin vers le fichier Excel.
        column : str
            Nom de la colonne à utiliser.
        embedding_dim : int
            Dimension d'embedding.
        delay : int
            Délai de reconstruction.
        **kwargs
            Paramètres supplémentaires pour le Dataset.

        Returns
        -------
        Dataset
            Instance du dataset initialisé.
        """
        df = pd.read_excel(path)
        series = df[column].values.astype(float)
        logger.info(f"Série chargée depuis {path} ({len(series)} points, colonne '{column}')")
        return cls(series, embedding_dim=embedding_dim, delay=delay, **kwargs)
