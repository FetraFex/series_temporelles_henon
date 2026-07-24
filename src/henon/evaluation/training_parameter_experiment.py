"""
Expérimentation sur les paramètres d'entraînement.

Ce module implémente la classe TrainingParameterExperiment qui évalue
l'influence des paramètres d'apprentissage (taux d'apprentissage,
nombre d'epochs, tolérance d'erreur) sur la précision de prédiction
du MLP 6-8-1 avec activation Sigmoid.
"""

from __future__ import annotations

import logging
import time
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


class TrainingParameterExperiment:
    """
    Expérimentation pour optimiser les paramètres d'entraînement.

    Évalue l'influence du taux d'apprentissage, du nombre maximum
    d'epochs et de la tolérance d'erreur sur les performances du MLP.

    Architecture fixe : 6-8-1 (Sigmoid cachée, Linéaire en sortie).

    Parameters
    ----------
    dataset : Dataset
        Dataset préparé (normalisé et splitté).
    n_repetitions : int
        Nombre de répétitions par configuration.
    seed_base : int
        Graine de base pour la reproductibilité.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_repetitions: int = 10,
        seed_base: int = 42,
    ) -> None:
        self.dataset = dataset
        self.n_repetitions = n_repetitions
        self.seed_base = seed_base
        self.evaluator = Evaluation()
        self.loss_fn = MSELoss()

    # ------------------------------------------------------------------
    # Entraînement unitaire
    # ------------------------------------------------------------------

    def _train_single(
        self,
        repetition: int,
        learning_rate: float = 0.01,
        epochs: int = 500,
        patience: int = 30,
        tolerance: float = 1e-5,
    ) -> dict:
        """
        Entraîne un réseau une seule fois avec les paramètres donnés.

        Parameters
        ----------
        repetition : int
            Numéro de la répétition (pour la graine).
        learning_rate : float
            Taux d'apprentissage.
        epochs : int
            Nombre maximum d'epochs.
        patience : int
            Patience pour l'early stopping.
        tolerance : float
            Tolérance d'erreur pour la convergence.

        Returns
        -------
        dict
            Métriques, nombre d'epochs et temps d'entraînement.
        """
        seed = self.seed_base + repetition

        # Création du réseau 6-8-1 (Sigmoid cachée, Linéaire en sortie)
        network = Network(
            layer_sizes=[6, 8, 1],
            activations=["sigmoid", "linear"],
            seed=seed,
        )

        # Entraînement avec mesure du temps
        trainer = BackPropagation(
            network=network,
            loss=self.loss_fn,
            learning_rate=learning_rate,
        )

        start_time = time.perf_counter()
        history = trainer.train(
            X_train=self.dataset.X_train,
            y_train=self.dataset.y_train,
            X_val=self.dataset.X_val,
            y_val=self.dataset.y_val,
            epochs=epochs,
            patience=patience,
            tolerance=tolerance,
            verbose=False,
        )
        elapsed = time.perf_counter() - start_time

        n_epochs_trained = len(history["train_loss"])

        # Prédiction sur le test
        y_pred_norm = network.predict(self.dataset.X_test)

        # Dénormalisation pour calculer les métriques réelles
        y_pred = self.dataset.denormalize_y(y_pred_norm)
        y_true = self.dataset.denormalize_y(self.dataset.y_test)

        # Calcul des métriques
        metrics = self.evaluator.compute_all(y_true.flatten(), y_pred.flatten())
        metrics["n_epochs"] = n_epochs_trained
        metrics["train_time"] = elapsed
        metrics["train_loss_curve"] = history["train_loss"]

        logger.debug(
            f"  rep={repetition}: MSE={metrics['mse']:.8f}, "
            f"epochs={n_epochs_trained}, time={elapsed:.3f}s"
        )

        return metrics

    # ------------------------------------------------------------------
    # Expérimentation 1 : Taux d'apprentissage
    # ------------------------------------------------------------------

    def run_learning_rate(
        self,
        learning_rates: list[float] | None = None,
        epochs: int = 500,
        patience: int = 30,
        tolerance: float = 1e-5,
    ) -> pd.DataFrame:
        """
        Évalue l'influence du taux d'apprentissage.

        Parameters
        ----------
        learning_rates : list[float] | None
            Liste des taux d'apprentissage à tester.
        epochs : int
            Nombre maximum d'epochs (fixe).
        patience : int
            Patience pour l'early stopping (fixe).
        tolerance : float
            Tolérance d'erreur (fixe).

        Returns
        -------
        pd.DataFrame
            Tableau des résultats.
        """
        learning_rates = learning_rates or [0.001, 0.005, 0.01, 0.05, 0.1]

        logger.info(
            f"Expérimentation Learning Rate : "
            f"LR ∈ {learning_rates}, {self.n_repetitions} répétitions"
        )

        results = []

        for lr in learning_rates:
            logger.info(f"Test learning_rate = {lr}")
            lr_results = self._evaluate_parameter(
                param_name="learning_rate",
                param_value=lr,
                learning_rate=lr,
                epochs=epochs,
                patience=patience,
                tolerance=tolerance,
            )
            results.append(lr_results)

        df = pd.DataFrame(results)
        logger.info("Expérimentation Learning Rate terminée")
        return df

    # ------------------------------------------------------------------
    # Expérimentation 2 : Nombre maximum d'epochs
    # ------------------------------------------------------------------

    def run_epochs(
        self,
        epochs_list: list[int] | None = None,
        learning_rate: float = 0.01,
        patience: int = 30,
        tolerance: float = 1e-5,
    ) -> pd.DataFrame:
        """
        Évalue l'influence du nombre maximum d'epochs.

        Parameters
        ----------
        epochs_list : list[int] | None
            Liste des nombres d'epochs à tester.
        learning_rate : float
            Taux d'apprentissage (fixe, meilleur trouvé).
        patience : int
            Patience pour l'early stopping (fixe).
        tolerance : float
            Tolérance d'erreur (fixe).

        Returns
        -------
        pd.DataFrame
            Tableau des résultats.
        """
        epochs_list = epochs_list or [500, 1000, 2000, 5000]

        logger.info(
            f"Expérimentation Epochs : "
            f"epochs ∈ {epochs_list}, {self.n_repetitions} répétitions"
        )

        results = []

        for max_epochs in epochs_list:
            logger.info(f"Test max_epochs = {max_epochs}")
            epoch_results = self._evaluate_parameter(
                param_name="max_epochs",
                param_value=max_epochs,
                learning_rate=learning_rate,
                epochs=max_epochs,
                patience=patience,
                tolerance=tolerance,
            )
            results.append(epoch_results)

        df = pd.DataFrame(results)
        logger.info("Expérimentation Epochs terminée")
        return df

    # ------------------------------------------------------------------
    # Expérimentation 3 : Tolérance d'erreur
    # ------------------------------------------------------------------

    def run_tolerance(
        self,
        tolerances: list[float] | None = None,
        learning_rate: float = 0.01,
        epochs: int = 500,
        patience: int = 30,
    ) -> pd.DataFrame:
        """
        Évalue l'influence de la tolérance d'erreur.

        Parameters
        ----------
        tolerances : list[float] | None
            Liste des tolérances à tester.
        learning_rate : float
            Taux d'apprentissage (fixe, meilleur trouvé).
        epochs : int
            Nombre maximum d'epochs (fixe).
        patience : int
            Patience pour l'early stopping (fixe).

        Returns
        -------
        pd.DataFrame
            Tableau des résultats.
        """
        tolerances = tolerances or [1e-2, 1e-3, 1e-4, 1e-5]

        logger.info(
            f"Expérimentation Tolerance : "
            f"tol ∈ {tolerances}, {self.n_repetitions} répétitions"
        )

        results = []

        for tol in tolerances:
            logger.info(f"Test tolerance = {tol}")
            tol_results = self._evaluate_parameter(
                param_name="tolerance",
                param_value=tol,
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                tolerance=tol,
            )
            results.append(tol_results)

        df = pd.DataFrame(results)
        logger.info("Expérimentation Tolerance terminée")
        return df

    # ------------------------------------------------------------------
    # Évaluation générique
    # ------------------------------------------------------------------

    def _evaluate_parameter(
        self,
        param_name: str,
        param_value: float | int,
        learning_rate: float,
        epochs: int,
        patience: int,
        tolerance: float,
    ) -> dict:
        """
        Évalue une configuration de paramètre sur plusieurs répétitions.

        Parameters
        ----------
        param_name : str
            Nom du paramètre évalué.
        param_value : float | int
            Valeur du paramètre.
        learning_rate : float
            Taux d'apprentissage.
        epochs : int
            Nombre maximum d'epochs.
        patience : int
            Patience pour l'early stopping.
        tolerance : float
            Tolérance d'erreur.

        Returns
        -------
        dict
            Résultats (moyennes + écarts-types) pour cette configuration.
        """
        mse_list = []
        rmse_list = []
        mae_list = []
        nmse_list = []
        epochs_list = []
        time_list = []
        loss_curves = []

        for rep in range(self.n_repetitions):
            metrics = self._train_single(
                repetition=rep,
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                tolerance=tolerance,
            )
            mse_list.append(metrics["mse"])
            rmse_list.append(metrics["rmse"])
            mae_list.append(metrics["mae"])
            nmse_list.append(metrics["nmse"])
            epochs_list.append(metrics["n_epochs"])
            time_list.append(metrics["train_time"])
            loss_curves.append(metrics["train_loss_curve"])

        result = {
            param_name: param_value,
            "avg_mse": np.mean(mse_list),
            "avg_rmse": np.mean(rmse_list),
            "avg_mae": np.mean(mae_list),
            "avg_nmse": np.mean(nmse_list),
            "std_mse": np.std(mse_list),
            "std_rmse": np.std(rmse_list),
            "std_mae": np.std(mae_list),
            "std_nmse": np.std(nmse_list),
            "avg_epochs": np.mean(epochs_list),
            "avg_time": np.mean(time_list),
        }

        return result

    # ------------------------------------------------------------------
    # Identification du meilleur résultat
    # ------------------------------------------------------------------

    def compare(self, results_df: pd.DataFrame, param_col: str) -> dict:
        """
        Identifie la meilleure configuration.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.
        param_col : str
            Nom de la colonne du paramètre évalué.

        Returns
        -------
        dict
            Meilleure configuration et ses métriques.
        """
        best_idx = results_df["avg_mse"].idxmin()
        best = results_df.iloc[best_idx]

        return {
            "param_value": best[param_col],
            "avg_mse": best["avg_mse"],
            "avg_rmse": best["avg_rmse"],
            "avg_mae": best["avg_mae"],
            "avg_nmse": best["avg_nmse"],
            "avg_epochs": best["avg_epochs"],
            "avg_time": best["avg_time"],
        }

    # ------------------------------------------------------------------
    # Graphiques
    # ------------------------------------------------------------------

    def plot_results(
        self,
        lr_results: pd.DataFrame,
        epochs_results: pd.DataFrame,
        tol_results: pd.DataFrame,
        output_dir: str | Path,
    ) -> None:
        """
        Génère tous les graphiques de comparaison.

        Parameters
        ----------
        lr_results : pd.DataFrame
            Résultats de l'expérimentation Learning Rate.
        epochs_results : pd.DataFrame
            Résultats de l'expérimentation Epochs.
        tol_results : pd.DataFrame
            Résultats de l'expérimentation Tolerance.
        output_dir : str | Path
            Répertoire de sortie.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._plot_param_vs_mse(
            lr_results, "learning_rate", "Taux d'apprentissage",
            "Learning Rate vs MSE", output_dir / "learning_rate_plot",
        )
        self._plot_param_vs_mse(
            epochs_results, "max_epochs", "Nombre max d'epochs",
            "Epochs vs MSE", output_dir / "epochs_plot",
        )
        self._plot_param_vs_mse(
            tol_results, "tolerance", "Tolérance d'erreur",
            "Tolerance vs MSE", output_dir / "tolerance_plot",
        )

        self._plot_loss_curves(
            lr_results, "learning_rate", "Taux d'apprentissage",
            output_dir / "loss_curves_lr",
        )
        self._plot_training_time(
            lr_results, epochs_results, tol_results, output_dir,
        )

        logger.info(f"Graphiques sauvegardés dans {output_dir}")

    def _plot_param_vs_mse(
        self,
        results_df: pd.DataFrame,
        param_col: str,
        x_label: str,
        title: str,
        filepath_stem: Path,
    ) -> None:
        """
        Génère un graphique paramètre vs MSE.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.
        param_col : str
            Nom de la colonne du paramètre.
        x_label : str
            Label de l'axe X.
        title : str
            Titre du graphique.
        filepath_stem : Path
            Chemin de base sans extension.
        """
        best = self.compare(results_df, param_col)

        fig = go.Figure()

        # Bande d'erreur (écart-type)
        fig.add_trace(go.Scatter(
            x=results_df[param_col].tolist(),
            y=(results_df["avg_mse"] + results_df["std_mse"]).tolist(),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=results_df[param_col].tolist(),
            y=(results_df["avg_mse"] - results_df["std_mse"]).clip(lower=0).tolist(),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            showlegend=False,
            hoverinfo="skip",
        ))

        # Courbe principale
        fig.add_trace(go.Scatter(
            x=results_df[param_col].tolist(),
            y=results_df["avg_mse"].tolist(),
            mode="lines+markers",
            name="MSE moyen",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
        ))

        # Marqueur du meilleur résultat
        fig.add_trace(go.Scatter(
            x=[best["param_value"]],
            y=[best["avg_mse"]],
            mode="markers",
            name=f"Meilleur ({best['param_value']})",
            marker=dict(
                color="red",
                size=14,
                symbol="star",
                line=dict(width=2, color="darkred"),
            ),
        ))

        # Ligne verticale en pointillés
        fig.add_vline(
            x=best["param_value"],
            line_dash="dash",
            line_color="red",
            line_width=1,
        )

        # Annotation
        fig.add_annotation(
            x=best["param_value"],
            y=best["avg_mse"],
            text=(
                f"{best['param_value']}<br>"
                f"MSE = {best['avg_mse']:.8f}"
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
                    f"{title}<br>"
                    "<sup>Architecture 6-8-1 — Sigmoid</sup>"
                ),
                font=dict(family="Times New Roman, serif", size=16),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title=x_label,
                showgrid=True,
                gridwidth=0.5,
                gridcolor="rgba(0,0,0,0.1)",
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

        fig.write_image(str(filepath_stem) + ".png", scale=3)
        fig.write_image(str(filepath_stem) + ".svg")
        fig.write_html(str(filepath_stem) + ".html")

    def _plot_loss_curves(
        self,
        results_df: pd.DataFrame,
        param_col: str,
        x_label: str,
        filepath_stem: Path,
    ) -> None:
        """
        Génère les courbes de perte d'entraînement.

        Parameters
        ----------
        results_df : pd.DataFrame
            Tableau des résultats.
        param_col : str
            Nom de la colonne du paramètre.
        x_label : str
            Label de la légende.
        filepath_stem : Path
            Chemin de base sans extension.
        """
        fig = go.Figure()

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        for i, row in results_df.iterrows():
            curve = row.get("train_loss_curve", None)
            if curve is None or not isinstance(curve, list):
                continue
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=list(range(1, len(curve) + 1)),
                y=curve,
                mode="lines",
                name=f"{x_label} = {row[param_col]}",
                line=dict(color=color, width=1.5),
            ))

        fig.update_layout(
            title=dict(
                text=(
                    "Courbes de perte d'entraînement<br>"
                    "<sup>Architecture 6-8-1 — Sigmoid</sup>"
                ),
                font=dict(family="Times New Roman, serif", size=16),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="Epoch",
                showgrid=True,
                gridwidth=0.5,
                gridcolor="rgba(0,0,0,0.1)",
                showline=True,
                linewidth=1,
                linecolor="black",
            ),
            yaxis=dict(
                title="MSE (perte d'entraînement)",
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

        fig.write_image(str(filepath_stem) + ".png", scale=3)
        fig.write_image(str(filepath_stem) + ".svg")
        fig.write_html(str(filepath_stem) + ".html")

    def _plot_training_time(
        self,
        lr_results: pd.DataFrame,
        epochs_results: pd.DataFrame,
        tol_results: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """
        Génère le graphique comparatif des temps d'entraînement.

        Parameters
        ----------
        lr_results : pd.DataFrame
            Résultats Learning Rate.
        epochs_results : pd.DataFrame
            Résultats Epochs.
        tol_results : pd.DataFrame
            Résultats Tolerance.
        output_dir : Path
            Répertoire de sortie.
        """
        fig = go.Figure()

        # Groupes de barres
        categories = ["Learning Rate", "Epochs", "Tolerance"]
        avg_times = [
            lr_results["avg_time"].mean(),
            epochs_results["avg_time"].mean(),
            tol_results["avg_time"].mean(),
        ]

        fig.add_trace(go.Bar(
            x=categories,
            y=avg_times,
            name="Temps moyen (s)",
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            text=[f"{t:.3f}s" for t in avg_times],
            textposition="outside",
            textfont=dict(size=11),
        ))

        fig.update_layout(
            title=dict(
                text=(
                    "Comparaison des temps d'entraînement<br>"
                    "<sup>Architecture 6-8-1 — Sigmoid</sup>"
                ),
                font=dict(family="Times New Roman, serif", size=16),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title="Expérimentation",
                showgrid=False,
                showline=True,
                linewidth=1,
                linecolor="black",
            ),
            yaxis=dict(
                title="Temps moyen (secondes)",
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

        filepath_stem = output_dir / "training_time_plot"
        fig.write_image(str(filepath_stem) + ".png", scale=3)
        fig.write_image(str(filepath_stem) + ".svg")
        fig.write_html(str(filepath_stem) + ".html")

    # ------------------------------------------------------------------
    # Export Excel
    # ------------------------------------------------------------------

    def export_results(
        self,
        lr_results: pd.DataFrame,
        epochs_results: pd.DataFrame,
        tol_results: pd.DataFrame,
        output_dir: str | Path,
    ) -> None:
        """
        Exporte les résultats des trois expérimentations en Excel.

        Parameters
        ----------
        lr_results : pd.DataFrame
            Résultats Learning Rate.
        epochs_results : pd.DataFrame
            Résultats Epochs.
        tol_results : pd.DataFrame
            Résultats Tolerance.
        output_dir : str | Path
            Répertoire de sortie.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Nettoyage des colonnes de courbes avant export
        lr_export = lr_results.drop(columns=["train_loss_curve"], errors="ignore")
        epochs_export = epochs_results.drop(columns=["train_loss_curve"], errors="ignore")
        tol_export = tol_results.drop(columns=["train_loss_curve"], errors="ignore")

        lr_export.to_excel(
            output_dir / "learning_rate_results.xlsx",
            index=False, float_format="%.8f",
        )
        epochs_export.to_excel(
            output_dir / "epochs_results.xlsx",
            index=False, float_format="%.8f",
        )
        tol_export.to_excel(
            output_dir / "tolerance_results.xlsx",
            index=False, float_format="%.8f",
        )

        logger.info(f"Résultats exportés dans {output_dir}")

    # ------------------------------------------------------------------
    # Rapport final
    # ------------------------------------------------------------------

    def print_best(
        self,
        lr_results: pd.DataFrame,
        epochs_results: pd.DataFrame,
        tol_results: pd.DataFrame,
    ) -> None:
        """
        Affiche le rapport final des meilleurs paramètres.

        Parameters
        ----------
        lr_results : pd.DataFrame
            Résultats Learning Rate.
        epochs_results : pd.DataFrame
            Résultats Epochs.
        tol_results : pd.DataFrame
            Résultats Tolerance.
        """
        best_lr = self.compare(lr_results, "learning_rate")
        best_epochs = self.compare(epochs_results, "max_epochs")
        best_tol = self.compare(tol_results, "tolerance")

        # Recalculer le meilleur MSE global en combinant les meilleurs paramètres
        # (on prend le meilleur de chaque expérimentation)
        best_mse = min(best_lr["avg_mse"], best_epochs["avg_mse"], best_tol["avg_mse"])

        print("\n" + "=" * 55)
        print("  BEST LEARNING PARAMETERS")
        print("=" * 55)
        print(f"  Learning Rate    : {best_lr['param_value']}")
        print(f"  Maximum Epochs   : {int(best_epochs['param_value'])}")
        print(f"  Tolerance        : {best_tol['param_value']}")
        print("-" * 55)
        print(f"  Average MSE      : {best_lr['avg_mse']:.8f}")
        print(f"  Average RMSE     : {best_lr['avg_rmse']:.8f}")
        print(f"  Average MAE      : {best_lr['avg_mae']:.8f}")
        print(f"  Average NMSE     : {best_lr['avg_nmse']:.8f}")
        print(f"  Average Epochs   : {best_lr['avg_epochs']:.1f}")
        print(f"  Average Time     : {best_lr['avg_time']:.4f}s")
        print("=" * 55 + "\n")
