from pathlib import Path

import pandas as pd

from henon import generate_henon_series

OUTPUT_DIR = Path("data/raw")


def main() -> None:
    X, Y = generate_henon_series(iterations=500, x0=0.0, y0=0.0, a=1.4, b=0.3)

    df = pd.DataFrame({
        "n": range(len(X)),
        "Xn": X,
        "Yn": Y,
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = OUTPUT_DIR / "henon_500.xlsx"
    df.to_excel(xlsx_path, index=False, sheet_name="Hénon")

    print(f"{len(df)} valeurs sauvegardées dans {xlsx_path}")


if __name__ == "__main__":
    main()