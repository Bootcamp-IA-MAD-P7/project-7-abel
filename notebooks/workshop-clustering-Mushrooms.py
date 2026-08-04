# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Taller de Aprendizaje No Supervisado
# ## Parte 1: Dataset de Setas (variables categóricas)

# %% [markdown]
# ### Mushroom Dataset
#
# Podéis obtener el conjunto de datos en el siguiente enlace:
#
# [Mushroom Dataset](https://www.kaggle.com/uciml/mushroom-classification)
#
# Como podréis comprobar, hay **muchas variables, todas categóricas**, por lo que las exploraciones con *scatterplot* no nos serán útiles como en otros casos.
#
# La variable a predecir es `class` (`e` = comestible / *edible*, `p` = venenosa / *poisonous*) y es **binaria**.
#
# > En este taller usaremos las etiquetas **solo para validar** lo que descubre el clustering. La idea del aprendizaje no supervisado es encontrar estructura *sin* mirar la etiqueta.

# %% [markdown]
# ### Algoritmos que cubriremos
#
# **Reducción de dimensionalidad:** PCA (lineal) y t-SNE (no lineal).
#
# **Clustering:** K-Means, Clustering Jerárquico (Aglomerativo), Gaussian Mixture Models (GMM) y DBSCAN.
#
# **Evaluación:** método del codo, *silhouette*, Davies-Bouldin, Calinski-Harabasz y (como tenemos etiqueta) Adjusted Rand Index y NMI.
#
# **Detección de anomalías:** Isolation Forest.

# %%
# Carga de librerias, las que hemos considerado basicas, anadid lo que querais :)
# %matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Importad aqui el resto de algoritmos a medida que los necesiteis:
# TSNE, AgglomerativeClustering, DBSCAN, GaussianMixture, IsolationForest, metricas...
sns.set_theme(style='whitegrid')
RANDOM_STATE = 42

# %% [markdown]
# ### Leer conjunto de datos y primer vistazo

# %%
# Leer el csv (esta en 'data/raw/mushrooms.csv') y sacar por pantalla las cinco primeras filas.

# %% [markdown]
# ### Exploración de datos

# %%
# Descripcion del conjunto de datos, estandar.

# %%
# Informacion sobre el tipo de datos de cada feature.

# %% [markdown]
# #### Calcular el número de nulos de cada feature

# %%
# Igual que otras veces, una linea: contar los nulos por variable.

# %% [markdown]
# #### Buscar valores extraños. Para ello, ver los valores únicos en cada feature

# %%
# Obtener un nuevo dataframe: en una columna las features (feature)
# y en la otra el numero de valores unicos asociados (n_values).
# n_values = ...

# %% [markdown]
# Observad dos cosas:
# - `veil-type` tiene **un único valor** → no aporta información.
# - `stalk-root` contiene el valor `'?'`, que en realidad es un **valor desconocido (nulo encubierto)**.

# %% [markdown]
# #### Tratar aquellos valores que entendamos que sean nulos

# %%
# Imputaciones. Podeis quitar esos puntos (fila entera), imputar con la moda
# o dejar ese valor como una posibilidad mas. Aqui imputamos '?' con la moda.

# %% [markdown]
# #### ¿Todas las features aportan información? Si alguna no aporta, eliminadla

# %%
# Dejar por el camino las features con un solo valor (no aportan nada).

# %% [markdown]
# #### Separar entre variables predictoras y variable a predecir

# %%
# La variable a predecir es 'class'.
# y =
# X =

# %% [markdown]
# #### Codificar correctamente las variables categóricas a numéricas

# %%
# One Hot Encoder (una linea). Pista: pd.get_dummies

# %% [markdown]
# #### Train / test split

# %%
# Os lo dejamos a todos igual
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %% [markdown]
# ## PCA
#
# Es un conjunto de datos del que aún no hemos visto gráficas, así que vamos a hacer algunas. Tenemos muchas variables, **PCA al rescate**: le pedimos dos dimensiones y las pintamos. Serán **las que retengan más información (varianza)**.

# %%
pca =       # metodo de sklearn
pca.fit(X_train)

# Representar en un scatterplot y poner en color las etiquetas de entrenamiento

# %% [markdown]
# Parece que está bastante separadito, ¡a ojo mucho se puede ver! :)

# %% [markdown]
# Antes de seguir, entrenamos un clasificador supervisado como **línea base** (así sabemos cuánta información hay realmente en los datos).

# %%
# 1. Definir el clasificador y el numero de estimadores
# 2. Entrenar en train
# 3. Calcular la precision sobre test

# %% [markdown]
# Es un conjunto sencillo y Random Forest es muy bueno. Veamos cuántas features tenemos:

# %%
X_train.shape

# %% [markdown]
# ¿Muchas features, no? Vamos a reducirlas con PCA y ver cuántas componentes necesita Random Forest para mantener su precisión.

# %%
n_features = # definir un rango de valores a probar
scores = []

for n in n_features:
    # 1. Definir y ajustar PCA sobre X_train
    # 2. Entrenar Random Forest sobre los datos reducidos
    # 3. Guardar el score en test

sns.lineplot(x=list(n_features), y=scores)

# %% [markdown]
# A partir de ~10 componentes ya tenemos la precisión que queríamos, reduciendo las variables a una fracción de las originales.

# %% [markdown]
# ---
# ## t-SNE: reducción **no lineal** para visualizar
#
# PCA es lineal. **t-SNE** intenta preservar la vecindad local y suele separar mejor los grupos visualmente. Es más caro, así que lo calculamos sobre una **muestra**.

# %%
# 1. Tomar una muestra de X_train (t-SNE es lento)
# 2. Definir TSNE(n_components=2, ...) y ajustar
# 3. Pintar el embedding coloreado por la etiqueta

# %% [markdown]
# ---
# ## Clustering
#
# El conjunto es sencillito, así que probemos clustering para ver qué información obtenemos **sin usar las etiquetas**. Trabajaremos sobre una representación reducida con PCA (10 componentes), que limpia ruido y acelera los algoritmos.

# %%
# Reducir X (one-hot completo) a 10 componentes con PCA -> X_red

# %% [markdown]
# ### K-Means: ¿cuántos clusters? Codo + Silhouette
#
# El **método del codo** mira la inercia (suma de distancias a los centroides). El **coeficiente de silhouette** mide cómo de bien separados están los clusters (cuanto más alto, mejor). Usamos ambos.

# %%
from sklearn.cluster import KMeans

k_values = # definir un rango
inercias, silhouettes = [], []
for k in k_values:
    # Definir KMeans y ajustar sobre X_red
    # Guardar la inercia (km.inertia_) y el silhouette_score

# Pintar las dos curvas (codo y silhouette)

# %% [markdown]
# ### K-Means final y comparación con la etiqueta
#
# Sabemos que hay dos clases (comestible / venenosa), así que probamos `k=2`. Con `catplot` vemos la distribución de la etiqueta real dentro de cada cluster.

# %%
# Aprender KMeans con el k obtenido y preparar el catplot.
kmeans = # Definir y entrenar KMeans

# ax = sns.catplot(col=, x=, data=, kind='count', col_wrap=4)
# Calcular ARI y NMI frente a y_bin

# %% [markdown]
# > **ARI / NMI** comparan los clusters con la etiqueta real (0 = aleatorio, 1 = idéntico). Sin haber visto la etiqueta, K-Means recupera buena parte de la estructura comestible/venenosa, pero **no es perfecto**: ese es el reto real del no supervisado.

# %% [markdown]
# ### Comparativa de algoritmos de clustering
#
# Vamos a poner a competir **K-Means, Aglomerativo, GMM y DBSCAN** con varias métricas. Las tres primeras métricas son *internas* (no usan etiqueta); ARI sí la usa, para validar.

# %%
# Definir una funcion evaluar(nombre, labels, X) que devuelva silhouette,
# davies_bouldin, calinski_harabasz y ARI.
# Aplicarla a KMeans, AgglomerativeClustering, GaussianMixture y DBSCAN
# y montar una tabla (DataFrame) comparativa.

# %% [markdown]
# ### Dendrograma (clustering jerárquico)
#
# El clustering aglomerativo construye una jerarquía que podemos visualizar como **dendrograma**. La altura a la que se unen dos grupos indica cómo de distintos son. Lo calculamos sobre una muestra para que se lea bien.

# %%
# 1. Tomar una muestra de X_red
# 2. linkage(..., method='ward')
# 3. dendrogram(...) para visualizar la jerarquia

# %% [markdown]
# ### DBSCAN: la **métrica de distancia importa**
#
# DBSCAN agrupa por densidad. Pero con datos **categóricos** codificados en one-hot, la distancia euclídea no captura bien la similitud. Comparemos euclídea (sobre PCA) con la distancia de **Jaccard** (pensada para datos binarios).

# %%
# Comparar DBSCAN con metric='euclidean' (sobre X_red) y metric='jaccard'
# (sobre el one-hot binario). Observar cual recupera mejor la estructura (ARI).

# %% [markdown]
# > **Lección:** con datos categóricos, elegir la distancia adecuada (Jaccard/Hamming) puede cambiar por completo el resultado de un algoritmo basado en densidad. No hay un algoritmo que gane siempre: depende del tipo de datos.

# %% [markdown]
# ### Visualización final: clusters vs etiqueta real
#
# Repetimos el scatter PCA, pero coloreando por el cluster de K-Means y por la etiqueta real, lado a lado.

# %%
# Entrenar PCA(2) sobre X para representar.
# Pintar dos scatter: uno coloreado por el cluster de KMeans y otro por la etiqueta real.

# %% [markdown]
# Es bastante parecido, ¿no? No es tan bueno como Random Forest (que usa etiquetas), pero K-Means ha identificado bastante bien la estructura **sin usarlas**. Si no tuviéramos etiquetas, esta aproximación nos ayudaría mucho a clasificar los tipos de hongos.

# %% [markdown]
# ---
# ## Detección de anomalías (Isolation Forest)
#
# Una tarea no supervisada distinta: encontrar las muestras **atípicas**. Isolation Forest aísla los puntos raros con pocos cortes aleatorios.

# %%
# 1. Definir IsolationForest(contamination=...) y ajustar sobre X_red
# 2. predict -> -1 son anomalias
# 3. Pintarlas sobre la proyeccion PCA

# %% [markdown]
# ---
# ## Para ir más allá (opcional)
#
# - **UMAP** (`pip install umap-learn`): alternativa a t-SNE, más rápida y preserva mejor la estructura global.
# - **HDBSCAN** (`pip install hdbscan`): DBSCAN jerárquico que no necesita fijar `eps`.
# - **Reglas de asociación** (`mlxtend`, Apriori/FP-Growth): muy naturales aquí por ser datos categóricos; permiten descubrir reglas tipo «si olor = X entonces venenosa».
#
# ## Conclusiones
#
# - PCA y t-SNE nos dejaron *ver* un dataset de >100 dimensiones.
# - K-Means, GMM y Aglomerativo recuperan la estructura comestible/venenosa con ARI ~0.6 **sin usar la etiqueta**.
# - DBSCAN nos enseñó que **la métrica de distancia importa** con datos categóricos.
# - Isolation Forest localiza las setas más atípicas.
# - El no supervisado no da una respuesta «perfecta», pero descubre estructura muy útil cuando no tenemos etiquetas.
