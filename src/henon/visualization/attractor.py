"""
Visualisation de l'attracteur de Hénon.

Ce module fournit des fonctions pour tracer et exporter
la représentation graphique yₙ = f(xₙ) du système de Hénon
sous différents styles professionnels à l'aide de Plotly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Constantes internes
# ---------------------------------------------------------------------------

STYLES_VALIDES = (
    "scientific",
    "publication",
    "gradient",
    "dark",
    "glow",
    "density",
    "trajectory",
    "premium",
)


# ---------------------------------------------------------------------------
# Fonctions utilitaires privées
# ---------------------------------------------------------------------------


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


def _validate_style(style: str) -> None:
    """
    Vérifie que le style demandé existe.

    Raises
    ------
    ValueError
        Si le style n'est pas reconnu.
    """
    if style not in STYLES_VALIDES:
        raise ValueError(
            f"Style inconnu '{style}'. "
            f"Styles disponibles : {', '.join(STYLES_VALIDES)}."
        )


def _ensure_output_dir(output_dir: Path) -> None:
    """Crée le dossier de sortie s'il n'existe pas."""
    output_dir.mkdir(parents=True, exist_ok=True)


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


def _apply_axis_defaults(
    fig: go.Figure,
    font_family: str = "Times New Roman, serif",
    font_size_title: int = 18,
    font_size_label: int = 14,
    title_text: str = "Attracteur de Hénon",
) -> None:
    """
    Applique la mise en page de base des axes et du titre.

    Cette fonction factorise la configuration commune à la plupart des styles.
    """
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(family=font_family, size=font_size_title),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(
                text="xₙ",
                font=dict(family=font_family, size=font_size_label),
            ),
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
        ),
        yaxis=dict(
            title=dict(
                text="yₙ",
                font=dict(family=font_family, size=font_size_label),
            ),
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
        ),
        width=800,
        height=600,
        margin=dict(l=60, r=30, t=60, b=60),
    )


# ---------------------------------------------------------------------------
# Styles graphiques privés
# ---------------------------------------------------------------------------


def _style_scientific(
    x: list[float],
    y: list[float],
    *,
    color: str,
    marker_size: float,
    opacity: float,
    **_: object,
) -> go.Figure:
    """
    Style scientifique (IEEE).

    Fond blanc, bleu scientifique, marqueurs fins, grille discrète.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color=color, size=marker_size, opacity=opacity),
        showlegend=False,
    ))
    _apply_axis_defaults(fig)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def _style_publication(
    x: list[float],
    y: list[float],
    *,
    color: str,
    marker_size: float,
    opacity: float,
    **_: object,
) -> go.Figure:
    """
    Style publication (Springer / Elsevier).

    Fond blanc, marqueurs noirs, minimaliste, axes épais.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color=color, size=marker_size, opacity=opacity),
        showlegend=False,
    ))
    _apply_axis_defaults(fig, font_size_title=16, font_size_label=12)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, linewidth=1.5, linecolor="black"),
        yaxis=dict(showgrid=False, linewidth=1.5, linecolor="black"),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def _style_gradient(
    x: list[float],
    y: list[float],
    *,
    marker_size: float,
    opacity: float,
    **_: object,
) -> go.Figure:
    """
    Style gradient temporel.

    Les points sont colorés selon leur indice temporel
    avec la palette Turbo.
    """
    n = len(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            size=marker_size,
            opacity=opacity,
            color=list(range(n)),
            colorscale="Turbo",
            colorbar=dict(
                title=dict(text="Indice n", font=dict(family="Times New Roman, serif")),
                thickness=15,
                len=0.75,
            ),
        ),
        showlegend=False,
    ))
    _apply_axis_defaults(fig)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def _style_dark(
    x: list[float],
    y: list[float],
    *,
    marker_size: float,
    opacity: float,
    **_: object,
) -> go.Figure:
    """
    Style sombre.

    Fond noir, palette cyan, grille discrète, police blanche.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color="#00e5ff", size=marker_size, opacity=opacity),
        showlegend=False,
    ))
    _apply_axis_defaults(fig, font_size_title=18, font_size_label=14)
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0e0e0e",
        paper_bgcolor="#0e0e0e",
        font=dict(color="white"),
        xaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor="rgba(255,255,255,0.1)",
            linecolor="white",
        ),
        yaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor="rgba(255,255,255,0.1)",
            linecolor="white",
        ),
    )
    return fig


def _style_glow(
    x: list[float],
    y: list[float],
    *,
    color: str,
    **_: object,
) -> go.Figure:
    """
    Style lumineux (glow).

    Superpose plusieurs couches de Scatter avec des tailles
    et opacités décroissantes pour créer un effet de halo.
    """
    fig = go.Figure()
    layers = [
        (12, 0.05),
        (7, 0.15),
        (3, 0.90),
    ]
    for size, alpha in layers:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(color=color, size=size, opacity=alpha),
            showlegend=False,
        ))
    _apply_axis_defaults(fig)
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0a0a1a",
        paper_bgcolor="#0a0a1a",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, linecolor="rgba(255,255,255,0.3)"),
        yaxis=dict(showgrid=False, linecolor="rgba(255,255,255,0.3)"),
    )
    return fig


def _style_density(
    x: list[float],
    y: list[float],
    **_: object,
) -> go.Figure:
    """
    Style densité.

    Utilise une Heatmap 2D pour visualiser les régions
    les plus fréquentées de l'attracteur.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=x, y=y,
        colorscale="Viridis",
        nbinsx=80,
        nbinsy=80,
        colorbar=dict(
            title=dict(text="Densité", font=dict(family="Times New Roman, serif")),
            thickness=15,
            len=0.75,
        ),
    ))
    _apply_axis_defaults(fig)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def _style_trajectory(
    x: list[float],
    y: list[float],
    *,
    color: str,
    marker_size: float,
    opacity: float,
    line_width: float,
    **_: object,
) -> go.Figure:
    """
    Style trajectoire.

    Relie les points par des lignes fines pour visualiser
    l'évolution temporelle du système.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(color=color, width=line_width),
        marker=dict(color=color, size=marker_size, opacity=opacity),
        showlegend=False,
    ))
    _apply_axis_defaults(fig)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def _style_premium(
    x: list[float],
    y: list[float],
    *,
    marker_size: float,
    opacity: float,
    **_: object,
) -> go.Figure:
    """
    Style premium (Apple / Tesla / OpenAI).

    Fond très clair, ombres légères, couleurs élégantes,
    typographie premium, mise en page équilibrée.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            color="#6366f1",
            size=marker_size,
            opacity=opacity,
            line=dict(width=0.3, color="rgba(99,102,241,0.3)"),
        ),
        showlegend=False,
    ))
    _apply_axis_defaults(fig, font_family="SF Pro Display, Helvetica Neue, Arial")
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
        xaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.04)",
            zeroline=False, linecolor="#d1d5db", linewidth=0.8,
        ),
        yaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.04)",
            zeroline=False, linecolor="#d1d5db", linewidth=0.8,
        ),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Registre des styles
# ---------------------------------------------------------------------------

_STYLE_BUILDERS: dict[str, callable] = {
    "scientific": _style_scientific,
    "publication": _style_publication,
    "gradient": _style_gradient,
    "dark": _style_dark,
    "glow": _style_glow,
    "density": _style_density,
    "trajectory": _style_trajectory,
    "premium": _style_premium,
}


# ---------------------------------------------------------------------------
# Fonction publique principale
# ---------------------------------------------------------------------------


def plot_henon_attractor(
    x_values: list[float],
    y_values: list[float],
    output_dir: str | Path = "results/figures",
    style: str = "scientific",
    color: str = "#1f77b4",
    marker_size: float = 1.5,
    opacity: float = 0.6,
    line_width: float = 0.5,
    width: int = 800,
    height: int = 600,
    font_family: str = "Times New Roman, serif",
    font_size_title: int = 18,
    font_size_label: int = 14,
    show_grid: bool = True,
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
    style : str
        Style graphique parmi : scientific, publication, gradient,
        dark, glow, density, trajectory, premium.
    color : str
        Code couleur des points (hex ou nom CSS).
    marker_size : float
        Taille des marqueurs.
    opacity : float
        Transparence des points (0.0 à 1.0).
    line_width : float
        Épaisseur des lignes (style trajectory).
    width : int
        Largeur de la figure en pixels.
    height : int
        Hauteur de la figure en pixels.
    font_family : str
        Police de caractères utilisée.
    font_size_title : int
        Taille de la police du titre.
    font_size_label : int
        Taille de la police des axes.
    show_grid : bool
        Si True, affiche la grille.
    show : bool
        Si True, affiche le graphique à l'écran.

    Returns
    -------
    go.Figure
        Objet figure Plotly configuré.

    Raises
    ------
    ValueError
        Si les données sont invalides ou le style inconnu.
    """
    _validate_inputs(x_values, y_values)
    _validate_style(style)

    # Construction de la figure via le builder du style choisi
    builder = _STYLE_BUILDERS[style]
    fig = builder(
        x_values, y_values,
        color=color,
        marker_size=marker_size,
        opacity=opacity,
        line_width=line_width,
    )

    # Application des dimensions
    fig.update_layout(width=width, height=height)

    # Export
    output_path = Path(output_dir) / style
    _ensure_output_dir(output_path)
    _export_figures(fig, output_path, f"henon_attractor_{style}")

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# Fonction de comparaison des styles
# ---------------------------------------------------------------------------


def compare_styles(
    x_values: list[float],
    y_values: list[float],
    output_dir: str | Path = "results/figures/styles",
    show: bool = False,
) -> dict[str, go.Figure]:
    """
    Génère automatiquement les huit styles et les enregistre.

    Parameters
    ----------
    x_values : list[float]
        Séquence des valeurs xₙ.
    y_values : list[float]
        Séquence des valeurs yₙ.
    output_dir : str | Path
        Dossier racine pour l'export de chaque style.
    show : bool
        Si True, affiche chaque figure à l'écran.

    Returns
    -------
    dict[str, go.Figure]
        Dictionnaire {nom_du_style: figure}.
    """
    _validate_inputs(x_values, y_values)

    figures: dict[str, go.Figure] = {}
    output_path = Path(output_dir)
    _ensure_output_dir(output_path)

    for style_name in STYLES_VALIDES:
        fig = plot_henon_attractor(
            x_values=x_values,
            y_values=y_values,
            output_dir=output_dir,
            style=style_name,
            show=show,
        )
        figures[style_name] = fig
        print(f"  Style '{style_name}' exporté dans {output_path / style_name}/")

    return figures
