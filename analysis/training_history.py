"""
Module d'analyse et de suivi de l'historique d'apprentissage.

Ce module fournit la classe TrainingHistory qui permet d'enregistrer,
d'exporter et de visualiser l'évolution de l'erreur pendant l'apprentissage
du réseau de neurones.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingHistory:
    """
    Enregistre et analyse l'historique d'apprentissage du réseau.

    Cette classe capture les métriques à chaque époque et fournit
    des méthodes d'exportation (CSV, TXT) et de visualisation (PNG).

    Parameters
    ----------
    output_dir : str | Path
        Répertoire où seront sauvegardés les résultats.
    """

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Données d'historique
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # Métadonnées
        self.architecture: str = "6-8-1"
        self.learning_rate: float = 0.1
        self.max_epochs: int = 1000
        self.tolerance: float = 1e-5
        self.training_time: float = 0.0
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
    ) -> None:
        """
        Enregistre une époque d'apprentissage.

        Parameters
        ----------
        epoch : int
            Numéro de l'époque (1-indexé).
        train_loss : float
            Erreur MSE sur l'ensemble d'apprentissage.
        val_loss : float | None
            Erreur MSE sur l'ensemble de validation.
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
        else:
            self.val_losses.append(float("nan"))

    def set_metadata(
        self,
        architecture: str = "6-8-1",
        learning_rate: float = 0.1,
        max_epochs: int = 1000,
        tolerance: float = 1e-5,
        training_time: float = 0.0,
    ) -> None:
        """
        Définit les métadonnées de l'expérience.

        Parameters
        ----------
        architecture : str
            Architecture du réseau (ex: "6-8-1").
        learning_rate : float
            Taux d'apprentissage utilisé.
        max_epochs : int
            Nombre maximum d'époques.
        tolerance : float
            Tolérance d'erreur.
        training_time : float
            Temps total d'entraînement en secondes.
        """
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.training_time = training_time

    @property
    def initial_loss(self) -> float:
        """Retourne la première erreur d'apprentissage."""
        return self.train_losses[0] if self.train_losses else float("inf")

    @property
    def final_loss(self) -> float:
        """Retourne la dernière erreur d'apprentissage."""
        return self.train_losses[-1] if self.train_losses else float("inf")

    @property
    def total_epochs(self) -> int:
        """Retourne le nombre total d'époques."""
        return len(self.epochs)

    def export_csv(self) -> Path:
        """
        Exporte l'historique en fichier CSV.

        Returns
        -------
        Path
            Chemin du fichier CSV généré.
        """
        filepath = self.output_dir / "training_history.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "validation_loss"])
            for i in range(len(self.epochs)):
                writer.writerow([
                    self.epochs[i],
                    f"{self.train_losses[i]:.10f}",
                    f"{self.val_losses[i]:.10f}"
                    if not (self.val_losses[i] != self.val_losses[i])
                    else "",
                ])
        return filepath

    def export_summary(self) -> Path:
        """
        Exporte un résumé texte de l'apprentissage.

        Returns
        -------
        Path
            Chemin du fichier texte généré.
        """
        filepath = self.output_dir / "training_summary.txt"
        lines = [
            "",
            "=" * 40,
            "TRAINING SUMMARY",
            "=" * 40,
            "",
            f"Architecture : {self.architecture}",
            "",
            f"Initial loss : {self.initial_loss:.10f}",
            "",
            f"Final loss : {self.final_loss:.10f}",
            "",
            f"Best validation loss : {self.best_val_loss:.10f}",
            "",
            f"Epoch best model : {self.best_epoch}",
            "",
            f"Total epochs : {self.total_epochs}",
            "",
            f"Training time : {self.training_time:.4f}s",
            "",
            "=" * 40,
            "",
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return filepath

    def plot_loss_curve(self) -> Path:
        """
        Génère le graphique de la courbe de perte.

        Returns
        -------
        Path
            Chemin du fichier PNG généré.
        """
        filepath = self.output_dir / "loss_curve.png"

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            self.epochs,
            self.train_losses,
            color="#1f77b4",
            linewidth=1.5,
            label="Apprentissage",
        )

        # Tracer la validation seulement si des valeurs valides existent
        val_epochs = [
            self.epochs[i]
            for i in range(len(self.val_losses))
            if self.val_losses[i] == self.val_losses[i]
        ]
        val_losses = [v for v in self.val_losses if v == v]

        if val_epochs:
            ax.plot(
                val_epochs,
                val_losses,
                color="#ff7f0e",
                linewidth=1.5,
                label="Validation",
            )

        ax.set_title(
            "Évolution de l'erreur durant l'apprentissage\n"
            f"du réseau {self.architecture}",
            fontsize=14,
            fontfamily="serif",
        )
        ax.set_xlabel("Nombre d'époques", fontsize=12)
        ax.set_ylabel("Erreur quadratique moyenne (MSE)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def plot_loss_zoom(self, n_last: int = 20) -> Path:
        """
        Génère un graphique zoomé sur les dernières époques.

        Parameters
        ----------
        n_last : int
            Nombre de dernières époques à afficher.

        Returns
        -------
        Path
            Chemin du fichier PNG généré.
        """
        filepath = self.output_dir / "loss_zoom_final_epochs.png"

        n = min(n_last, len(self.epochs))
        start_idx = len(self.epochs) - n

        epochs_zoom = self.epochs[start_idx:]
        train_zoom = self.train_losses[start_idx:]
        val_zoom = self.val_losses[start_idx:]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            epochs_zoom,
            train_zoom,
            color="#1f77b4",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Apprentissage",
        )

        val_epochs_zoom = [
            epochs_zoom[i]
            for i in range(len(val_zoom))
            if val_zoom[i] == val_zoom[i]
        ]
        val_losses_zoom = [v for v in val_zoom if v == v]

        if val_epochs_zoom:
            ax.plot(
                val_epochs_zoom,
                val_losses_zoom,
                color="#ff7f0e",
                linewidth=1.5,
                marker="s",
                markersize=4,
                label="Validation",
            )

        ax.set_title(
            "Convergence finale du réseau neuronal\n"
            f"Dernières {n} époques — Architecture {self.architecture}",
            fontsize=14,
            fontfamily="serif",
        )
        ax.set_xlabel("Nombre d'époques", fontsize=12)
        ax.set_ylabel("Erreur quadratique moyenne (MSE)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def export_all(self) -> list[Path]:
        """
        Exporte tous les fichiers (CSV, TXT, PNG).

        Returns
        -------
        list[Path]
            Liste des chemins de fichiers générés.
        """
        paths = [
            self.export_csv(),
            self.export_summary(),
            self.plot_loss_curve(),
            self.plot_loss_zoom(),
        ]
        return paths
