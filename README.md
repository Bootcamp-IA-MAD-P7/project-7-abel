# 🧠 Taller de Aprendizaje Automático **No Supervisado**

Este repositorio contiene un taller práctico de **machine learning no supervisado**: reducción de dimensionalidad (**PCA**, **t-SNE**), **clustering** (K-Means, Aglomerativo, GMM, DBSCAN) y **detección de anomalías** (Isolation Forest).

El taller se divide en **dos notebooks complementarios**, cada uno con su propio dataset. La gracia está en el **contraste entre ambos**: un mismo conjunto de técnicas se comporta de forma muy distinta según el tipo de datos y según si tenemos o no una etiqueta de referencia.

- 🍄 **Parte 1 — Setas** (`mushrooms.csv`): datos **categóricos**, *con* etiqueta (`class`).
- 💳 **Parte 2 — Tarjetas de crédito** (`credit_card.csv`): datos **numéricos**, *sin* etiqueta.

---

## 📓 Acceso directo a los notebooks

| Notebook | Dataset | Tipo de datos | ¿Hay etiqueta? |
|---|---|---|---|
| [🍄 `workshop-clustering-Mushrooms.ipynb`](notebooks/workshop-clustering-Mushrooms.ipynb) | `data/raw/mushrooms.csv` | Categóricos | Sí — `class` (solo para **validar**) |
| [💳 `workshop-clustering-creditcard.ipynb`](notebooks/workshop-clustering-creditcard.ipynb) | `data/raw/credit_card.csv` | Numéricos | No — segmentación **de verdad** |

> Se entregan **los dos notebooks**. No son independientes: la Parte 2 da por sabido lo aprendido en la Parte 1.

---

## 🗂️ Estructura del repositorio

```
project-7-abel/
├── data/
│   ├── raw/          # Datos originales descargados (CSV)
│   ├── cleaned/      # Datos limpios generados (parquet)
│   └── processed/    # Datos listos para modelar (parquet)
├── notebooks/
│   ├── workshop-clustering-Mushrooms.py      # Fuente jupytext (Parte 1)
│   ├── workshop-clustering-Mushrooms.ipynb   # Notebook (Parte 1)
│   ├── workshop-clustering-creditcard.py     # Fuente jupytext (Parte 2)
│   └── workshop-clustering-creditcard.ipynb  # Notebook (Parte 2)
├── scripts/          # Utilidades (vacío por ahora)
├── main.py           # Orquestador: ejecuta los dos notebooks en secuencia
├── pyproject.toml    # Dependencias del proyecto (uv)
├── uv.lock           # Bloqueo de versiones (uv)
└── jupytext.toml     # Emparejamiento .py ↔ .ipynb en notebooks/
```

---

## 🍄 Parte 1 — Setas (datos categóricos, *con* etiqueta)

**Notebook:** [`workshop-clustering-Mushrooms.ipynb`](notebooks/workshop-clustering-Mushrooms.ipynb) · **Dataset:** [`data/raw/mushrooms.csv`](data/raw/mushrooms.csv)
🔗 [Mushroom Dataset (Kaggle)](https://www.kaggle.com/uciml/mushroom-classification) · [UCI](https://archive.ics.uci.edu/ml/datasets/Mushroom)

Cada fila es un hongo descrito con **~22 variables, todas categóricas** (forma, color, olor, etc.). La variable `class` es **binaria**: `e` (comestible) / `p` (venenoso).

La clave pedagógica: **tenemos etiqueta, pero el clustering NO la usa**. La reservamos *solo para validar* a posteriori cuánta estructura real ha recuperado el modelo sin haberla visto.

**Qué se implementa:**
- Carga y EDA: descripción, tipos, nulos y detección de nulos encubiertos (el valor `'?'`).
- Eliminación de columnas constantes (`veil-type`) e imputación con la moda.
- **One-Hot Encoding** (`pd.get_dummies`) → dataset de >100 dimensiones.
- **PCA** y **t-SNE** para visualizar los datos en 2D.
- **Random Forest** como *línea base supervisada* y estudio de cuántas componentes PCA bastan para mantener la precisión.
- **Clustering**: K-Means (método del codo + *silhouette*), Aglomerativo (con **dendrograma**), GMM y DBSCAN.
- Lección con **DBSCAN**: con datos categóricos one-hot, la **distancia importa** (euclídea vs **Jaccard**).
- Validación con etiqueta: **Adjusted Rand Index (ARI)** y **NMI**.
- **Isolation Forest** para detección de anomalías.

**Artefactos generados:**
- `data/cleaned/mushrooms_cleaned.parquet` (datos limpios)
- `data/processed/mushrooms_encoded.parquet` (one-hot, listo para modelar)

---

## 💳 Parte 2 — Tarjetas de crédito (datos numéricos, *sin* etiqueta)

**Notebook:** [`workshop-clustering-creditcard.ipynb`](notebooks/workshop-clustering-creditcard.ipynb) · **Dataset:** [`data/raw/credit_card.csv`](data/raw/credit_card.csv)
🔗 [Credit Card Dataset for Clustering (Kaggle)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

Comportamiento de uso de ~9.000 titulares de tarjeta durante 6 meses, con **17 variables numéricas** (saldo, compras, adelantos de efectivo, límite, pagos…).

La clave pedagógica: **aquí NO hay etiqueta**. Es aprendizaje no supervisado «de verdad»: no se puede calcular ARI porque no existe una verdad de referencia. El éxito se mide con **métricas internas** y, sobre todo, con la **interpretabilidad** de los segmentos. El objetivo es **segmentar clientes** para una estrategia de marketing.

**Qué se implementa:**
- Carga y EDA: tamaño, descripción, tipos y detección de nulos.
- Limpieza: eliminación de `CUST_ID` e imputación de nulos con la **mediana** (robusta ante el sesgo).
- Observación del **sesgo** típico de datos financieros (histogramas).
- **Escalado** (`StandardScaler`) — imprescindible con variables de escalas muy distintas.
- **PCA**: varianza explicada acumulada (*scree plot*) y proyección a 2D (los datos son una **nube continua**).
- **Clustering**: K-Means (codo + *silhouette*), Aglomerativo (dendrograma), GMM y DBSCAN.
- Validación **sin etiqueta**: *silhouette*, *Davies-Bouldin* y *Calinski-Harabasz*.
- Visualización de los segmentos con **t-SNE**.
- **Interpretación de perfiles** (heatmap de medias por cluster) → nombrar los segmentos en términos de negocio (VIP, riesgo, poco activos…). **Este es el entregable del caso.**
- **Isolation Forest** para detectar clientes atípicos.

**Artefactos generados:**
- `data/cleaned/credit_card_cleaned.parquet` (datos limpios)
- `data/processed/credit_card_scaled.parquet` (escalados, listos para modelar)

---

## 🔄 Pipeline de datos

```
data/raw/        ──►  data/cleaned/        ──►  data/processed/
(CSV originales)      (parquet limpios)         (parquet listos para modelar)
```

Los notebooks leen los CSV desde `data/raw/` y guardan los resultados intermedios como **parquet** (vía `pyarrow`) en `data/cleaned/` y `data/processed/`. Las rutas de guardado son **relativas al notebook**, así que funcionan desde cualquier directorio de trabajo.

---

## 📦 Instalación

El proyecto usa **uv** como gestor de dependencias y **Python ≥ 3.12**.

### Opción A — Con uv (recomendado)

```bash
# 1. Instalar uv si no lo tienes (una vez por máquina)
pip install uv
# o bien: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clonar el repositorio
git clone <url-del-repositorio>
cd project-7-abel

# 3. Crear el entorno e instalar dependencias (incluye el grupo dev: jupyter + jupytext)
uv sync --extra dev
# o simplemente:
uv sync
```

### Opción B — Con pip y un entorno virtual clásico

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn pyarrow streamlit
pip install jupyter jupytext     # grupo dev
```

---

## 🚀 Cómo ejecutar

### 1. Notebooks (interactivo)

```bash
uv run jupyter notebook
```

Abre cualquiera de los dos notebooks en `notebooks/` (o ábrelos directamente en **VS Code**). Ejecuta las celdas en orden. Las celdas marcadas con comentarios (`# ...`) traen pistas, no la solución: están pensadas para **completarse durante el taller**.

### 2. Scripts (ejecución directa)

Cada notebook también existe como script Python (formato **jupytext percent**), ejecutable de principio a fin:

```bash
# Parte 1 — Setas
uv run python notebooks/workshop-clustering-Mushrooms.py

# Parte 2 — Tarjetas de crédito
uv run python notebooks/workshop-clustering-creditcard.py
```

> En Windows, si no quieres usar `uv run`, activa el entorno antes: `.venv\Scripts\activate && python notebooks/workshop-clustering-Mushrooms.py`.

### 3. Orquestador (todo en secuencia)

```bash
uv run python main.py
```

Ejecuta **los dos scripts en orden**, regenera los parquet de `data/cleaned/` y `data/processed/` y muestra las gráficas al final.

### 4. Sincronización jupytext (`.py` ↔ `.ipynb`)

Los pares `.py` y `.ipynb` están **emparejados** por jupytext (ver `jupytext.toml`). Cualquier cambio en uno debe sincronizarse con el otro:

```bash
# Tras editar el .py, regenerar el .ipynb
uv run jupytext --sync notebooks/workshop-clustering-Mushrooms.py

# O sincronizar todos los notebooks de la carpeta
uv run jupytext --sync notebooks/
```

---

## 📥 Descarga de los datasets

Los CSV ya están incluidos en [`data/raw/`](data/raw/). Si necesitas descargarlos de nuevo:

| Dataset | Fuente | Ruta destino |
|---|---|---|
| Mushroom Classification | [Kaggle](https://www.kaggle.com/uciml/mushroom-classification) / [UCI](https://archive.ics.uci.edu/ml/datasets/Mushroom) | `data/raw/mushrooms.csv` |
| Credit Card Dataset for Clustering | [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) | `data/raw/credit_card.csv` |

Coloca el CSV descargado con ese nombre exacto en `data/raw/`. Los notebooks los leen desde ahí.

---

## 🧩 ¿Por qué dos datasets?

| | 🍄 Setas | 💳 Tarjetas |
|---|---|---|
| Variables | Categóricas | Numéricas |
| Preprocesado clave | One-Hot Encoding | Escalado / imputación |
| Etiqueta | Sí (solo validar) | **No** |
| Cómo se valida | ARI / NMI (vs etiqueta) | Métricas internas + interpretabilidad |
| Estructura | Grupos separables | Nube continua |
| Distancia | Jaccard > euclídea | Euclídea sobre datos escalados |

**Conclusión transversal:** no hay un algoritmo ni una métrica que gane siempre. El acierto está en elegir el preprocesado, la distancia y la forma de validar **según el tipo de datos y el problema**.

---

## 🔧 Tecnologías

- Python ≥ 3.12 · Pandas · NumPy · PyArrow (parquet)
- Seaborn · Matplotlib
- Scikit-learn: `PCA`, `TSNE`, `KMeans`, `AgglomerativeClustering`, `GaussianMixture`, `DBSCAN`, `IsolationForest`, `RandomForestClassifier` y métricas de clustering
- SciPy (`linkage` / `dendrogram`)
- Jupyter · Jupytext (emparejamiento `.py` ↔ `.ipynb`)
- *(Opcional, para ir más allá)* `umap-learn`, `hdbscan`, `mlxtend`, Streamlit

---

## 📊 Evaluación

Se evaluarán las siguientes competencias **en ambos notebooks**:

**Competencia: Evaluar conjuntos de datos con herramientas de análisis y visualización**
- ✅ Uso y gestión de formato `.csv` y `.parquet`
- ✅ Limpieza y preprocesado de datos
- ✅ Visualización de datos (Seaborn, Matplotlib)
- ✅ Análisis exploratorio detallado (EDA)
- ✅ Técnicas de preprocesado (normalización, escalado, one-hot encoding)
- ✅ Técnicas avanzadas de limpieza (atípicos, imputación de faltantes)
- ✅ Técnicas de reducción de dimensionalidad (PCA, t-SNE)

**Competencia: Aplicar algoritmos de ML según el problema**
- ✅ Seleccionar las variables útiles y descartar las que no aportan
- ✅ Reconocer un caso de aprendizaje no supervisado
- ✅ Aplicar modelos de clustering
- ✅ Distinguir regresión / clasificación / clustering
- ✅ Separación de datos en train/test (Parte 1)
- ✅ Uso de modelos de *ensemble* (RandomForest como baseline en la Parte 1)
- ✅ Interpretación y validación de los resultados (ARI/NMI y métricas internas)

---

> 💡 Cada notebook termina con una sección **«Para ir más allá»** con extensiones opcionales (UMAP, HDBSCAN, reglas de asociación, ingeniería de KPIs…) para quien quiera profundizar.
