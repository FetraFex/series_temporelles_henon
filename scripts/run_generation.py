from pathlib import Path

import pandas as pd

from henon import generate_henon_series


def main() -> None:
    """Génère la série de Hénon et sauvegarde en Excel."""

    # Génération des 500 itérations du système de Hénon
    X, Y = generate_henon_series(iterations=500, x0=0.0, y0=0.0, a=1.4, b=0.3)

    # Sauvegarde en fichier Excel
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "n": range(len(X)),
        "Xn": X,
        "Yn": Y,
    })

    xlsx_path = raw_dir / "henon_500.xlsx"
    df.to_excel(xlsx_path, index=False, sheet_name="Hénon")
    print(f"{len(df)} valeurs sauvegardées dans {xlsx_path}")


if __name__ == "__main__":
    main()
