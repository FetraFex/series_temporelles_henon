"""
Visualisation de l'attracteur de Hénon.

Ce module fournit une fonction pour tracer et exporter
la représentation graphique yₙ = f(xₙ) du système de Hénon
à l'aide de Plotly.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def _validate_inputs(
    x_values: list[float],
    y_values: list[float],
) -> None:
    """
    Valide la cohérence des données d'entrée.

    Raises
    ------
    ValueError
        Si les listes sont de tailles différentes ou vides.
    """
    if len(x_values) != len(y_values):
        raise ValueError(
            f"Les listes doivent avoir la même taille : "
            f"x_values ({len(x_values)}) ≠ y_values ({len(y_values)})."
        )
    if len(x_values) == 0:
        raise ValueError(
            "Les listes x_values et y_values ne doivent pas être vides."
        )


def _export_figures(fig: go.Figure, output_path: Path, name: str) -> None:
    """
    Exporte la figure en PNG (haute résolution), SVG et PDF.

    Parameters
    ----------
    fig : go.Figure
        Figure Plotly à exporter.
    output_path : Path
        Dossier cible.
    name : str
        Nom de base du fichier (sans extension).
    """
    fig.write_image(str(output_path / f"{name}.png"), scale=3)
    fig.write_image(str(output_path / f"{name}.svg"))
    fig.write_image(str(output_path / f"{name}.pdf"))


def plot_henon_attractor(
    x_values: list[float],
    y_values: list[float],
    output_dir: str | Path = "results/figures",
    color: str = "#1f77b4",
    marker_size: float = 3.5,
    opacity: float = 0.6,
    width: int = 800,
    height: int = 600,
    show: bool = True,
) -> go.Figure:
    """
    Trace et exporte l'attracteur de Hénon : yₙ = f(xₙ).

    Parameters
    ----------
    x_values : list[float]
        Séquence des valeurs xₙ du système de Hénon.
    y_values : list[float]
        Séquence des valeurs yₙ du système de Hénon.
    output_dir : str | Path
        Répertoire de sortie pour les figures exportées.
    color : str
        Code couleur des points (hex ou nom CSS).
    marker_size : float
        Taille des marqueurs.
    opacity : float
        Transparence des points (0.0 à 1.0).
    width : int
        Largeur de la figure en pixels.
    height : int
        Hauteur de la figure en pixels.
    show : bool
        Si True, affiche le graphique à l'écran.

    Returns
    -------
    go.Figure
        Objet figure Plotly configuré.

    Raises
    ------
    ValueError
        Si les données sont invalides.
    """
    _validate_inputs(x_values, y_values)

    # Construction du scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers",
        marker=dict(color=color, size=marker_size, opacity=opacity),
        showlegend=False,
    ))

    # Mise en page académique
    fig.update_layout(
        title=dict(
            text="Attracteur de Hénon",
            font=dict(family="Times New Roman, serif", size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="xₙ", font=dict(family="Times New Roman, serif", size=14)),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
        ),
        yaxis=dict(
            title=dict(text="yₙ", font=dict(family="Times New Roman, serif", size=14)),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
        ),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    # Export
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _export_figures(fig, output_path, "henon_attractor_scientific")

    if show:
        fig.show()

    return fig
