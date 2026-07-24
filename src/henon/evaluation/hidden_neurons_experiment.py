"""
Expérimentation sur le nombre de neurones cachés.

Ce module implémente la classe HiddenNeuronExperiment qui évalue
différentes architectures 6-H-1 pour déterminer le nombre optimal
de neurones cachés.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from henon.core.neural.activations import get_activation
from henon.core.neural.backprop import BackPropagation
from henon.core.neural.losses import MSELoss
from henon.core.neural.network import Network
from henon.data.supervised import Dataset
from henon.evaluation.metrics import Evaluation

logger = logging.getLogger(__name__)


class HiddenNeuronExperiment:
    """
    Expérimentation pour déterminer le nombre optimal de neurones cachés.

    Évalue les architectures 6-H-1 pour H ∈ [hidden_range],
    avec n_repetitions répétitions par architecture pour tenir compte
    de l'aléatoire de l'initialisation des poids.

    Parameters
    ----------
    dataset : Dataset
        Dataset préparé (normalisé et splitté).
    hidden_range : list[int]
        Liste des nombres de neurones cachés à tester.
    n_repetitions : int
        Nombre de répétitions par architecture.
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
        hidden_range: list[int] | None = None,
        n_repetitions: int = 10,
        learning_rate: float = 0.01,
        epochs: int = 500,
        patience: int = 30,
        seed_base: int = 42,
    ) -> None:
        self.dataset = dataset
        self.hidden_range = hidden_range or [6, 7, 8, 9, 10]
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
            f"H ∈ {self.hidden_range}, {self.n_repetitions} répétitions"
        )

        results = []

        for n_hidden in self.hidden_range:
            logger.info(f"Test architecture 6-{n_hidden}-1")
            arch_results = self.evaluate_architecture(n_hidden)
            results.append(arch_results)

        # Création du DataFrame récapitulatif
        df = pd.DataFrame(results)

        logger.info("Expérimentation terminée")
        return df

    def evaluate_architecture(self, n_hidden: int) -> dict:
        """
        Évalue une architecture donnée avec plusieurs répétitions.

        Parameters
        ----------
        n_hidden : int
            Nombre de neurones cachés.

        Returns
        -------
        dict
            Résultats (moyennes + écarts-types) pour cette architecture.
        """
        mse_list = []
        rmse_list = []
        mae_list = []
        nmse_list = []

        for rep in range(self.n_repetitions):
            metrics = self._train_single(n_hidden, rep)
            mse_list.append(metrics["mse"])
            rmse_list.append(metrics["rmse"])
            mae_list.append(metrics["mae"])
            nmse_list.append(metrics["nmse"])

        return {
            "hidden_neurons": n_hidden,
            "avg_mse": np.mean(mse_list),
            "avg_rmse": np.mean(rmse_list),
            "avg_mae": np.mean(mae_list),
            "avg_nmse": np.mean(nmse_list),
            "std_mse": np.std(mse_list),
            "std_rmse": np.std(rmse_list),
            "std_mae": np.std(mae_list),
            "std_nmse": np.std(nmse_list),
        }

    def _train_single(self, n_hidden: int, repetition: int) -> dict:
        """
        Entraîne un réseau une seule fois.

        Parameters
        ----------
        n_hidden : int
            Nombre de neurones cachés.
        repetition : int
            Numéro de la répétition (pour la graine).

        Returns
        -------
        dict
            Métriques sur le jeu de test.
        """
        seed = self.seed_base + repetition

        # Création du réseau 6-H-1
        network = Network(
            layer_sizes=[6, n_hidden, 1],
            activations=["tanh", "linear"],
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
            f"  6-{n_hidden}-1, rep={repetition}: "
            f"MSE={metrics['mse']:.6f}"
        )

        return metrics

    def compare_results(self, results_df: pd.DataFrame) -> dict:
        """
        Identifie la meilleure architecture.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.

        Returns
        -------
        dict
            Meilleure architecture et ses métriques.
        """
        best_idx = results_df["avg_mse"].idxmin()
        best = results_df.iloc[best_idx]

        return {
            "architecture": f"6-{int(best['hidden_neurons'])}-1",
            "hidden_neurons": int(best["hidden_neurons"]),
            "avg_mse": best["avg_mse"],
            "avg_rmse": best["avg_rmse"],
            "avg_mae": best["avg_mae"],
            "avg_nmse": best["avg_nmse"],
        }

    def plot_results(
        self,
        results_df: pd.ndarray,
        output_dir: str | Path,
    ) -> None:
        """
        Génère le graphique de comparaison des architectures.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.
        output_dir : str | Path
            Répertoire de sortie.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best = self.compare_results(results_df)

        # Graphique principal : MSE vs nombre de neurones cachés
        fig = go.Figure()

        # Barres d'erreur (écart-type)
        fig.add_trace(go.Scatter(
            x=results_df["hidden_neurons"].tolist(),
            y=(results_df["avg_mse"] + results_df["std_mse"]).tolist(),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=results_df["hidden_neurons"].tolist(),
            y=(results_df["avg_mse"] - results_df["std_mse"]).tolist(),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            showlegend=False,
            hoverinfo="skip",
        ))

        # Courbe principale
        fig.add_trace(go.Scatter(
            x=results_df["hidden_neurons"].tolist(),
            y=results_df["avg_mse"].tolist(),
            mode="lines+markers",
            name="MSE moyen",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
            error_y=dict(
                type="data",
                array=results_df["std_mse"].tolist(),
                visible=True,
            ),
        ))

        # Marqueur du meilleur résultat
        fig.add_trace(go.Scatter(
            x=[best["hidden_neurons"]],
            y=[best["avg_mse"]],
            mode="markers",
            name=f"Meilleur ({best['architecture']})",
            marker=dict(
                color="red",
                size=14,
                symbol="star",
                line=dict(width=2, color="darkred"),
            ),
        ))

        # Ligne verticale en pointillés
        fig.add_vline(
            x=best["hidden_neurons"],
            line_dash="dash",
            line_color="red",
            line_width=1,
        )

        # Annotation
        fig.add_annotation(
            x=best["hidden_neurons"],
            y=best["avg_mse"],
            text=(
                f"{best['architecture']}<br>"
                f"MSE = {best['avg_mse']:.6f}"
            ),
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            font=dict(size=11, color="red"),
        )

        fig.update_layout(
            title=dict(
                text=(
                    "Recherche du nombre optimal de neurones cachés<br>"
                    "<sup>Architecture 6-H-1 — Méthode ACP (Takens)</sup>"
                ),
                font=dict(family="Times New Roman, serif", size=16),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="Nombre de neurones cachés (H)",
                showgrid=True,
                gridwidth=0.5,
                gridcolor="rgba(0,0,0,0.1)",
                showline=True,
                linewidth=1,
                linecolor="black",
                dtick=1,
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
            legend=dict(
                font=dict(family="Times New Roman, serif", size=11),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
            ),
            width=800,
            height=500,
            margin=dict(l=60, r=30, t=80, b=60),
        )

        # Export
        fig.write_image(str(output_dir / "hidden_neurons_plot.png"), scale=3)
        fig.write_image(str(output_dir / "hidden_neurons_plot.svg"))
        fig.write_html(str(output_dir / "hidden_neurons_plot.html"))

        logger.info(f"Graphique sauvegardé dans {output_dir}")

    def export_results(
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

        filepath = output_dir / "hidden_neurons_results.xlsx"
        results_df.to_excel(filepath, index=False, float_format="%.10f")

        logger.info(f"Résultats exportés dans {filepath}")

    def print_best(self, results_df: pd.DataFrame) -> None:
        """
        Affiche la meilleure architecture en console.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.
        """
        best = self.compare_results(results_df)

        print("\n" + "=" * 50)
        print("  BEST ARCHITECTURE")
        print("=" * 50)
        print(f"  Architecture : {best['architecture']}")
        print(f"  Average MSE  : {best['avg_mse']:.6f}")
        print(f"  Average RMSE : {best['avg_rmse']:.6f}")
        print(f"  Average MAE  : {best['avg_mae']:.6f}")
        print(f"  Average NMSE : {best['avg_nmse']:.6f}")
        print("=" * 50 + "\n")
