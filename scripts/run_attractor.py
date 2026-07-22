"""Script de visualisation de l'attracteur de Hénon."""

from henon import generate_henon_series
from henon.visualization import plot_henon_attractor


def main() -> None:
    """Génère les données et trace l'attracteur."""

    X, Y = generate_henon_series(iterations=500, x0=0.0, y0=0.0, a=1.4, b=0.3)

    plot_henon_attractor(
        x_values=X,
        y_values=Y,
        output_dir="results/figures",
    )


if __name__ == "__main__":
    main()
