from henon import generate_henon_series
from henon.visualization import plot_henon_attractor


def main() -> None:
    """Trace et exporte l'attracteur de Hénon yₙ = f(xₙ)."""

    X, Y = generate_henon_series(iterations=500, x0=0.0, y0=0.0, a=1.4, b=0.3)

    plot_henon_attractor(
        x_values=X,
        y_values=Y,
        output_dir="results/figures",
    )
    print("Attracteur exporté dans results/figures/")


if __name__ == "__main__":
    main()
