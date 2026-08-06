"""Orquestador del taller de aprendizaje no supervisado.

Ejecuta de forma secuencial los dos notebooks (formato jupytext .py) que
resuelven el taller:

    1. notebooks/workshop-clustering-Mushrooms.py    (setas, categóricas)
    2. notebooks/workshop-clustering-creditcard.py   (tarjetas, numéricas)

Durante la ejecución se regeneran los artefactos intermedios en
``data/cleaned/`` y ``data/processed/`` como archivos .parquet.
"""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt

NOTEBOOKS = [
    "notebooks/workshop-clustering-Mushrooms.py",
    "notebooks/workshop-clustering-creditcard.py",
]


def run_workshop() -> None:
    """Ejecuta la lógica de los notebooks del taller y muestra las gráficas."""
    for notebook in NOTEBOOKS:
        path = Path(notebook)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el notebook: {path.resolve()}")
        print(f"\n=== Ejecutando {notebook} ===")
        runpy.run_path(str(path), run_name="__main__")

    plt.show()


if __name__ == "__main__":
    run_workshop()
