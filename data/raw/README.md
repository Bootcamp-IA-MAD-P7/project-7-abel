# Datos crudos (`raw`)

Conjuntos de datos **originales**, tal y como fueron descargados, **sin modificar**.

| Archivo | Contenido | Origen |
|---|---|---|
| `mushrooms.csv` | 8.124 setas descritas con ~22 variables categóricas y la etiqueta `class` (`e` = comestible, `p` = venenoso) | [Mushroom Dataset (Kaggle)](https://www.kaggle.com/uciml/mushroom-classification) / [UCI](https://archive.ics.uci.edu/ml/datasets/Mushroom) |
| `credit_card.csv` | ~9.000 titulares de tarjeta durante 6 meses, con 17 variables numéricas de uso | [Credit Card Dataset for Clustering (Kaggle)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) |

## Convención

- **No editar** estos archivos: son la fuente de referencia del proyecto.
- Los notebooks los leen directamente desde esta ruta:
  - Parte 1 (setas): `pd.read_csv("data/raw/mushrooms.csv")`
  - Parte 2 (tarjetas): `pd.read_csv("data/raw/credit_card.csv")`

Cualquier limpieza o transformación se realiza a partir de estos datos en los notebooks y, si se guarda el resultado, se escribe en `data/cleaned/` o `data/processed/`.
