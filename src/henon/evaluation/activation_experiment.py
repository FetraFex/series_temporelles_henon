"""
Expérimentation sur la fonction d'activation optimale.

Ce module implémente la classe ActivationExperiment qui évalue
les fonctions d'activation (Sigmoid, Tanh, Linear) pour la couche
cachée d'un MLP 6-8-1 et détermine laquelle offre les meilleures
performances sur la série de Hénon.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from henon.core.neural.backprop import BackPropagation
from henon.core.neural.losses import MSELoss
from henon.core.neural.network import Network
from henon.data.supervised import Dataset
from henon.evaluation.metrics import Evaluation

logger = logging.getLogger(__name__)


class ActivationExperiment:
    """
    Expérimentation pour déterminer la fonction d'activation optimale.

    Évalue Sigmoid, Tanh et Linear sur la couche cachée d'un MLP 6-8-1,
    avec n_repetitions répétitions par fonction pour tenir compte
    de l'aléatoire de l'initialisation des poids.

    Parameters
    ----------
    dataset : Dataset
        Dataset préparé (normalisé et splitté).
    activations : list[str]
        Liste des noms des fonctions d'activation à tester.
    n_repetitions : int
        Nombre de répétitions par activation.
    learning_rate : float
        Taux d'apprentissage.
    epochs : int
        Nombre maximum d'epochs.
    patience : int
        Patience pour l'early stopping.
    seed_base : int
        Graine de base pour la reproductibilité.
    """

    def __init__(
        self,
        dataset: Dataset,
        activations: list[str] | None = None,
        n_repetitions: int = 10,
        learning_rate: float = 0.01,
        epochs: int = 500,
        patience: int = 30,
        seed_base: int = 42,
    ) -> None:
        self.dataset = dataset
        self.activations = activations or ["sigmoid", "tanh", "linear"]
        self.n_repetitions = n_repetitions
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.seed_base = seed_base
        self.evaluator = Evaluation()
        self.loss_fn = MSELoss()

    def run(self) -> pd.DataFrame:
        """
        Exécute l'expérimentation complète.

        Returns
        -------
        pd.DataFrame
            Tableau des résultats avec moyennes et écarts-types.
        """
        logger.info(
            f"Démarrage de l'expérimentation : "
            f"activations = {self.activations}, "
            f"{self.n_repetitions} répétitions"
        )

        results = []

        for activation_name in self.activations:
            logger.info(f"Test activation : {activation_name}")
            act_results = self.evaluate(activation_name)
            results.append(act_results)

        df = pd.DataFrame(results)

        logger.info("Expérimentation terminée")
        return df

    def evaluate(self, activation_name: str) -> dict:
        """
        Évalue une fonction d'activation avec plusieurs répétitions.

        Parameters
        ----------
        activation_name : str
            Nom de la fonction d'activation ('sigmoid', 'tanh', 'linear').

        Returns
        -------
        dict
            Résultats (moyennes + écarts-types) pour cette activation.
        """
        mse_list = []
        rmse_list = []
        mae_list = []
        nmse_list = []

        for rep in range(self.n_repetitions):
            metrics = self._train_single(activation_name, rep)
            mse_list.append(metrics["mse"])
            rmse_list.append(metrics["rmse"])
            mae_list.append(metrics["mae"])
            nmse_list.append(metrics["nmse"])

        return {
            "activation": activation_name,
            "avg_mse": np.mean(mse_list),
            "avg_rmse": np.mean(rmse_list),
            "avg_mae": np.mean(mae_list),
            "avg_nmse": np.mean(nmse_list),
            "std_mse": np.std(mse_list),
            "std_rmse": np.std(rmse_list),
            "std_mae": np.std(mae_list),
            "std_nmse": np.std(nmse_list),
        }

    def _train_single(self, activation_name: str, repetition: int) -> dict:
        """
        Entraîne un réseau avec une activation donnée.

        Parameters
        ----------
        activation_name : str
            Nom de la fonction d'activation.
        repetition : int
            Numéro de la répétition (pour la graine).

        Returns
        -------
        dict
            Métriques sur le jeu de test.
        """
        seed = self.seed_base + repetition

        # Création du réseau 6-8-1 avec l'activation spécifiée
        network = Network(
            layer_sizes=[6, 8, 1],
            activations=[activation_name, "linear"],
            seed=seed,
        )

        # Entraînement
        trainer = BackPropagation(
            network=network,
            loss=self.loss_fn,
            learning_rate=self.learning_rate,
        )

        trainer.train(
            X_train=self.dataset.X_train,
            y_train=self.dataset.y_train,
            X_val=self.dataset.X_val,
            y_val=self.dataset.y_val,
            epochs=self.epochs,
            patience=self.patience,
            verbose=False,
        )

        # Prédiction sur le test
        y_pred_norm = network.predict(self.dataset.X_test)

        # Dénormalisation pour calculer les métriques réelles
        y_pred = self.dataset.denormalize_y(y_pred_norm)
        y_true = self.dataset.denormalize_y(self.dataset.y_test)

        # Calcul des métriques
        metrics = self.evaluator.compute_all(y_true.flatten(), y_pred.flatten())

        logger.debug(
            f"  activation={activation_name}, rep={repetition}: "
            f"MSE={metrics['mse']:.8f}"
        )

        return metrics

    def compare(self, results_df: pd.DataFrame) -> dict:
        """
        Identifie la meilleure fonction d'activation.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.

        Returns
        -------
        dict
            Meilleure activation et ses métriques.
        """
        best_idx = results_df["avg_mse"].idxmin()
        best = results_df.iloc[best_idx]

        return {
            "activation": best["activation"],
            "avg_mse": best["avg_mse"],
            "avg_rmse": best["avg_rmse"],
            "avg_mae": best["avg_mae"],
            "avg_nmse": best["avg_nmse"],
        }

    def plot(
        self,
        results_df: pd.DataFrame,
        output_dir: str | Path,
    ) -> None:
        """
        Génère le graphique de comparaison des activations.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.
        output_dir : str | Path
            Répertoire de sortie.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best = self.compare(results_df)

        # Couleurs : la meilleure activation est en rouge
        colors = []
        for act in results_df["activation"]:
            if act == best["activation"]:
                colors.append("#d62728")
            else:
                colors.append("#1f77b4")

        # Graphique en barres
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=results_df["activation"].tolist(),
            y=results_df["avg_mse"].tolist(),
            name="MSE moyen",
            marker_color=colors,
            error_y=dict(
                type="data",
                array=results_df["std_mse"].tolist(),
                visible=True,
            ),
            text=[f"{v:.8f}" for v in results_df["avg_mse"]],
            textposition="outside",
            textfont=dict(size=10),
        ))

        # Annotation pour la meilleure activation
        fig.add_annotation(
            x=best["activation"],
            y=best["avg_mse"],
            text=(
                f"Meilleure activation<br>"
                f"MSE = {best['avg_mse']:.8f}"
            ),
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(size=11, color="#d62728"),
        )

        fig.update_layout(
            title=dict(
                text=(
                    "Comparaison des fonctions d'activation<br>"
                    "<sup>Architecture 6-8-1 — Méthode ACP (Takens)</sup>"
                ),
                font=dict(family="Times New Roman, serif", size=16),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="Fonction d'activation",
                showgrid=False,
                showline=True,
                linewidth=1,
                linecolor="black",
            ),
            yaxis=dict(
                title="MSE moyen (± écart-type)",
                showgrid=True,
                gridwidth=0.5,
                gridcolor="rgba(0,0,0,0.1)",
                showline=True,
                linewidth=1,
                linecolor="black",
            ),
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            width=800,
            height=500,
            margin=dict(l=60, r=30, t=80, b=60),
        )

        # Export
        fig.write_image(str(output_dir / "activation_plot.png"), scale=3)
        fig.write_image(str(output_dir / "activation_plot.svg"))
        fig.write_html(str(output_dir / "activation_plot.html"))

        logger.info(f"Graphique sauvegardé dans {output_dir}")

    def export(
        self,
        results_df: pd.DataFrame,
        output_dir: str | Path,
    ) -> None:
        """
        Exporte les résultats en Excel.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.
        output_dir : str | Path
            Répertoire de sortie.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / "activation_results.xlsx"
        results_df.to_excel(filepath, index=False, float_format="%.8f")

        logger.info(f"Résultats exportés dans {filepath}")

    def print_best(self, results_df: pd.DataFrame) -> None:
        """
        Affiche la meilleure activation en console.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.
        """
        best = self.compare(results_df)

        print("\n" + "=" * 50)
        print("  BEST ACTIVATION FUNCTION")
        print("=" * 50)
        print(f"  Activation   : {best['activation']}")
        print(f"  Average MSE  : {best['avg_mse']:.8f}")
        print(f"  Average RMSE : {best['avg_rmse']:.8f}")
        print(f"  Average MAE  : {best['avg_mae']:.8f}")
        print(f"  Average NMSE : {best['avg_nmse']:.8f}")
        print("=" * 50 + "\n")
