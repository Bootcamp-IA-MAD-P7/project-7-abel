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
# ## Parte 2: Segmentación de Clientes de Tarjeta de Crédito (variables numéricas)

# %% [markdown]
# ### Credit Card Dataset
#
# [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
#
# Resume el comportamiento de uso de unos **9.000 titulares** de tarjeta durante los últimos 6 meses, con **17 variables numéricas de comportamiento** (saldo, compras, adelantos de efectivo, límite de crédito, pagos...).
#
# > **Diferencia clave con el dataset de setas:** aquí **no hay etiqueta**. El objetivo *es* encontrar segmentos de clientes para definir una estrategia de marketing. Esto es aprendizaje no supervisado «de verdad»: no podemos calcular ARI porque no hay verdad de referencia; nos guiamos por métricas internas y por la **interpretabilidad** de los segmentos.
#
# Además, al ser numérico y con escalas muy distintas, aparecen dos pasos que con las setas no hicieron falta: **imputar nulos** y **escalar**.

# %%
# %matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from scipy.cluster.hierarchy import linkage, dendrogram
sns.set_theme(style='whitegrid')
RANDOM_STATE = 42

# %% [markdown]
# ### Leer conjunto de datos y primer vistazo

# %%
# Leer el csv (esta en 'data/raw/credit_card.csv') y mostrar las primeras filas.
df = pd.read_csv('data/raw/credit_card.csv')
df.head()

# %% [markdown]
# ### Exploración de datos

# %%
# Tamano del dataset.
print(f'Tamaño del dataset: {df.shape[0]} filas y {df.shape[1]} columnas')

# %%
# Descripcion estadistica. Fijaos en las escalas tan distintas entre variables.
df.describe()

# %%
# Tipos de datos.
df.dtypes

# %% [markdown]
# #### Nulos

# %%
# Contar nulos por variable (mostrad solo las que tengan).
nulos = df.isnull().sum()
nulos[nulos > 0]

# %% [markdown]
# Hay nulos en `CREDIT_LIMIT` (1) y `MINIMUM_PAYMENTS` (~313). Al ser variables numéricas muy sesgadas, los imputamos con la **mediana** (más robusta que la media).

# %%
# 1. Eliminar CUST_ID (es un identificador)
df = df.drop(columns=['CUST_ID'])
# 2. Imputar los nulos con la mediana de cada columna
df = df.fillna(df.median())

# Guardamos el dataset limpio para reutilizarlo en fases posteriores
df.to_parquet('data/cleaned/credit_card_cleaned.parquet')

# %% [markdown]
# #### Distribución de algunas variables
#
# Muchas variables están **muy sesgadas** (la mayoría de clientes gasta poco y unos pocos muchísimo). Esto es típico en datos financieros.

# %%
# Pintar histogramas de unas cuantas variables (df[cols].hist...) y observar el sesgo.
cols_hist = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
df[cols_hist].hist(figsize=(14, 8), bins=30)
plt.suptitle('Histogramas de variables clave: se observa el sesgo típico financiero')
plt.tight_layout()

# %% [markdown]
# ### Escalado
#
# K-Means, PCA y casi todos los algoritmos de distancia son **sensibles a la escala**. `CREDIT_LIMIT` llega a miles y `PURCHASES_FREQUENCY` está entre 0 y 1: sin escalar, las variables grandes dominarían. Estandarizamos (media 0, desviación 1).

# %%
# Aplicar StandardScaler a df -> X (array escalado)
scaler = StandardScaler()
X = scaler.fit_transform(df)
X_scaled = pd.DataFrame(X, columns=df.columns)

# Guardamos el dataset escalado (listo para modelar) para reutilizarlo en fases posteriores
X_scaled.to_parquet('data/processed/credit_card_scaled.parquet')

# %% [markdown]
# ## PCA
#
# Con 17 variables no podemos pintar un scatter directo. Usamos PCA para (1) ver cuánta información retiene cada componente y (2) proyectar a 2D.

# %%
# 1. Ajustar PCA sin fijar n_components y mirar explained_variance_ratio_
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X)
print('Varianza explicada por componente:', pca_full.explained_variance_ratio_.round(3))
# 2. Pintar la varianza explicada acumulada (scree plot)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
plt.axhline(y=0.80, color='red', linestyle='--', label='80% de varianza')
plt.title('Varianza explicada acumulada (scree plot)')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.legend()
# 3. Decidir cuantas componentes hacen falta para ~80% de varianza
n_80 = np.argmax(cum_var >= 0.80) + 1
print(f'Hacen falta {n_80} componentes para explicar el 80% de la varianza')

# %%
# Proyectar X a 2 componentes y pintar el scatter (aun sin colores: no hay etiqueta)
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca_2d.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.4, s=10)
plt.title('Proyección PCA (2 componentes): nube continua de clientes')
plt.xlabel('PC1')
plt.ylabel('PC2')

# %% [markdown]
# A diferencia de las setas, aquí no vemos grupos separados a simple vista: es más bien una **nube continua**. El clustering nos ayudará a trazar fronteras útiles dentro de ella.

# %% [markdown]
# ## Clustering: ¿cuántos segmentos? Codo + Silhouette

# %%
k_values = range(2, 9)
inercias, silhouettes = [], []
for k in k_values:
    # Ajustar KMeans y guardar inercia + silhouette
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# Pintar las dos curvas y elegir best_k (el de mayor silhouette)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(list(k_values), inercias, marker='o')
ax[0].set_title('Método del codo (inercia)')
ax[0].set_xlabel('k')
ax[1].plot(list(k_values), silhouettes, marker='o', color='green')
ax[1].set_title('Coeficiente de silhouette')
ax[1].set_xlabel('k')
plt.tight_layout()

best_k = k_values[np.argmax(silhouettes)]
print(f'Mejor k según silhouette: {best_k} (silhouette = {max(silhouettes):.4f})')

# %% [markdown]
# ### K-Means final

# %%
# Entrenar KMeans con best_k, guardar las etiquetas y mirar el tamano de cada cluster
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(X)
labels_km = kmeans.labels_
print('Tamaño de cada cluster:', pd.Series(labels_km).value_counts().sort_index().to_dict())

# %% [markdown]
# ### Comparativa de algoritmos
#
# Sin etiqueta, comparamos con **métricas internas**: *silhouette* y *Calinski-Harabasz* (más alto = mejor) y *Davies-Bouldin* (más bajo = mejor).

# %%
# Definir evaluar(nombre, labels, X) con silhouette, davies_bouldin y calinski_harabasz
# (sin ARI: no hay etiqueta). Comparar KMeans, Aglomerativo y GMM.
def evaluar(nombre, labels, X):
    return {
        'modelo': nombre,
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
    }

modelos = {
    'KMeans': kmeans.labels_,
    'Aglomerativo': AgglomerativeClustering(n_clusters=best_k).fit_predict(X),
    'GMM': GaussianMixture(n_components=best_k, random_state=RANDOM_STATE).fit_predict(X),
}

tabla_comparativa = pd.DataFrame(
    [evaluar(nombre, labels, X) for nombre, labels in modelos.items()]
)
tabla_comparativa

# %% [markdown]
# ### Dendrograma

# %%
# linkage + dendrogram sobre una muestra de X
X_sample, _ = train_test_split(X, test_size=0.95, random_state=RANDOM_STATE)
Z = linkage(X_sample, method='ward')
plt.figure(figsize=(14, 6))
dendrogram(Z, no_labels=True)
plt.title('Dendrograma (linkage ward) sobre una muestra de X escalado')
plt.xlabel('Muestras')
plt.ylabel('Distancia')

# %% [markdown]
# ### DBSCAN: ¿hay clusters de densidad aquí?
#
# Probamos DBSCAN. Veréis que en alta dimensión tiende a juntar casi todo en **un solo cluster** y marcar el resto como **ruido**. Eso nos dice algo importante: estos datos son una nube continua, no grupos separados por densidad. Aquí DBSCAN funciona mejor como **detector de atípicos** que como segmentador.

# %%
# Ejecutar DBSCAN sobre la proyeccion PCA(2). Observar cuantos clusters y cuanto ruido.
dbscan = DBSCAN(eps=0.5, min_samples=10).fit(X_2d)
n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
n_ruido = (dbscan.labels_ == -1).sum()
print(f'DBSCAN (sobre PCA(2)) -> clusters: {n_clusters} | ruido: {n_ruido} '
      f'({n_ruido / len(dbscan.labels_) * 100:.1f}%)')

# %% [markdown]
# ### Visualización de los segmentos (t-SNE)
#
# Proyectamos con t-SNE (sobre una muestra) y coloreamos por el segmento de K-Means.

# %%
# t-SNE sobre una muestra de X, coloreado por el cluster de KMeans
X_tsne_sample, _, labels_tsne, _ = train_test_split(X, labels_km, test_size=0.8,
                                                    random_state=RANDOM_STATE, stratify=labels_km)
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_tsne_sample)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_tsne, palette='Set2', alpha=0.6)
plt.title('t-SNE (muestra) coloreado por el segmento de K-Means')
plt.legend(title='Cluster')

# %% [markdown]
# ## Interpretación de los segmentos
#
# Lo más importante en segmentación: **¿qué caracteriza a cada grupo?** Calculamos la media de cada variable por cluster y la estandarizamos entre clusters (rojo = por encima de la media, azul = por debajo). Así leemos el «perfil» de cada segmento.

# %%
# 1. Anadir la columna 'cluster' al df original (sin escalar)
df_clusters = df.copy()
df_clusters['cluster'] = labels_km

# 2. Calcular la media de cada variable por cluster
perfil_clusters = df_clusters.groupby('cluster').mean()
print(perfil_clusters)

# 3. Estandarizar entre clusters y pintar un heatmap (perfil de cada segmento)
perfil_z = perfil_clusters.apply(lambda fila: (fila - fila.mean()) / fila.std(), axis=1)
plt.figure(figsize=(14, 6))
sns.heatmap(perfil_z, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            cbar_kws={'label': 'Desviación respecto a la media entre clusters'})
plt.title('Perfil de cada segmento (medias estandarizadas entre clusters)')

# %% [markdown]
# Leyendo el heatmap se pueden nombrar los segmentos en términos de **negocio**, por ejemplo: clientes de alto saldo y muchas compras (VIP), clientes que tiran de adelantos de efectivo (riesgo), clientes poco activos, etc. Ese nombre y la estrategia asociada es justo el entregable que pide el caso.

# %%
# Nombrar cada segmento en terminos de negocio a partir de su perfil estandarizado
def nombre_segmento(fila):
    if fila['PURCHASES'] > 1:
        return 'VIP (grandes compradores)'
    if fila['CASH_ADVANCE'] > 1:
        return 'Riesgo (adelantos de efectivo)'
    if fila['CREDIT_LIMIT'] > 2:
        return 'Alto límite, bajo uso'
    return 'Perfil medio'

nombres_segmentos = perfil_z.apply(nombre_segmento, axis=1)
nombres_segmentos.index.name = 'cluster'
print(nombres_segmentos.to_string())

# %% [markdown]
# ## Detección de anomalías (Isolation Forest)
#
# Identificamos los clientes con comportamiento más atípico (posible fraude, errores de datos o clientes premium fuera de norma).

# %%
# IsolationForest sobre X, marcar los atipicos (-1) y pintarlos sobre PCA(2)
iso_forest = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
iso_forest.fit(X)
anomalias = iso_forest.predict(X)
print('Número de clientes atípicos:', (anomalias == -1).sum())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=(anomalias == -1),
                palette={False: 'blue', True: 'red'}, alpha=0.6, s=10)
plt.title('Clientes atípicos detectados por Isolation Forest (proyección PCA)')
plt.legend(title='Anomalía')

# %% [markdown]
# ---
# ## Para ir más allá (opcional)
#
# - **Ingeniería de KPIs**: derivar variables como *ratio de uso del límite* (`BALANCE / CREDIT_LIMIT`) o *compra media por transacción* suele mejorar mucho los segmentos.
# - **UMAP / HDBSCAN** para visualización y clustering por densidad más robusto.
# - **Transformación logarítmica** de las variables sesgadas antes de escalar.
#
# ## Conclusiones
#
# - Sin etiqueta, la segmentación se valida con métricas internas (*silhouette*, etc.) y, sobre todo, con la **interpretabilidad** de los perfiles.
# - Imputar y **escalar** fue imprescindible aquí (a diferencia de las setas).
# - K-Means, GMM y Aglomerativo dan segmentos coherentes; DBSCAN reveló que los datos son una nube continua (mejor como detector de atípicos).
# - El heatmap de perfiles convierte los clusters en **segmentos de negocio accionables**.
