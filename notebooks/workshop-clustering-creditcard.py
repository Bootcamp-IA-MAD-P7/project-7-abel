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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Importad el resto a medida que los necesiteis (TSNE, GMM, DBSCAN, metricas...)
sns.set_theme(style='whitegrid')
RANDOM_STATE = 42

# %% [markdown]
# ### Leer conjunto de datos y primer vistazo

# %%
# Leer el csv (esta en 'data/raw/credit_card.csv') y mostrar las primeras filas.

# %% [markdown]
# ### Exploración de datos

# %%
# Tamano del dataset.

# %%
# Descripcion estadistica. Fijaos en las escalas tan distintas entre variables.

# %%
# Tipos de datos.

# %% [markdown]
# #### Nulos

# %%
# Contar nulos por variable (mostrad solo las que tengan).

# %% [markdown]
# Hay nulos en `CREDIT_LIMIT` (1) y `MINIMUM_PAYMENTS` (~313). Al ser variables numéricas muy sesgadas, los imputamos con la **mediana** (más robusta que la media).

# %%
# 1. Eliminar CUST_ID (es un identificador)
# 2. Imputar los nulos con la mediana de cada columna

# %% [markdown]
# #### Distribución de algunas variables
#
# Muchas variables están **muy sesgadas** (la mayoría de clientes gasta poco y unos pocos muchísimo). Esto es típico en datos financieros.

# %%
# Pintar histogramas de unas cuantas variables (df[cols].hist...) y observar el sesgo.

# %% [markdown]
# ### Escalado
#
# K-Means, PCA y casi todos los algoritmos de distancia son **sensibles a la escala**. `CREDIT_LIMIT` llega a miles y `PURCHASES_FREQUENCY` está entre 0 y 1: sin escalar, las variables grandes dominarían. Estandarizamos (media 0, desviación 1).

# %%
# Aplicar StandardScaler a df -> X (array escalado)

# %% [markdown]
# ## PCA
#
# Con 17 variables no podemos pintar un scatter directo. Usamos PCA para (1) ver cuánta información retiene cada componente y (2) proyectar a 2D.

# %%
# 1. Ajustar PCA sin fijar n_components y mirar explained_variance_ratio_
# 2. Pintar la varianza explicada acumulada (scree plot)
# 3. Decidir cuantas componentes hacen falta para ~80% de varianza

# %%
# Proyectar X a 2 componentes y pintar el scatter (aun sin colores: no hay etiqueta)

# %% [markdown]
# A diferencia de las setas, aquí no vemos grupos separados a simple vista: es más bien una **nube continua**. El clustering nos ayudará a trazar fronteras útiles dentro de ella.

# %% [markdown]
# ## Clustering: ¿cuántos segmentos? Codo + Silhouette

# %%
k_values = range(2, 9)
inercias, silhouettes = [], []
for k in k_values:
    # Ajustar KMeans y guardar inercia + silhouette
    pass
# Pintar las dos curvas y elegir best_k (el de mayor silhouette)

# %% [markdown]
# ### K-Means final

# %%
# Entrenar KMeans con best_k, guardar las etiquetas y mirar el tamano de cada cluster

# %% [markdown]
# ### Comparativa de algoritmos
#
# Sin etiqueta, comparamos con **métricas internas**: *silhouette* y *Calinski-Harabasz* (más alto = mejor) y *Davies-Bouldin* (más bajo = mejor).

# %%
# Definir evaluar(nombre, labels, X) con silhouette, davies_bouldin y calinski_harabasz
# (sin ARI: no hay etiqueta). Comparar KMeans, Aglomerativo y GMM.

# %% [markdown]
# ### Dendrograma

# %%
# linkage + dendrogram sobre una muestra de X

# %% [markdown]
# ### DBSCAN: ¿hay clusters de densidad aquí?
#
# Probamos DBSCAN. Veréis que en alta dimensión tiende a juntar casi todo en **un solo cluster** y marcar el resto como **ruido**. Eso nos dice algo importante: estos datos son una nube continua, no grupos separados por densidad. Aquí DBSCAN funciona mejor como **detector de atípicos** que como segmentador.

# %%
# Ejecutar DBSCAN sobre la proyeccion PCA(2). Observar cuantos clusters y cuanto ruido.

# %% [markdown]
# ### Visualización de los segmentos (t-SNE)
#
# Proyectamos con t-SNE (sobre una muestra) y coloreamos por el segmento de K-Means.

# %%
# t-SNE sobre una muestra de X, coloreado por el cluster de KMeans

# %% [markdown]
# ## Interpretación de los segmentos
#
# Lo más importante en segmentación: **¿qué caracteriza a cada grupo?** Calculamos la media de cada variable por cluster y la estandarizamos entre clusters (rojo = por encima de la media, azul = por debajo). Así leemos el «perfil» de cada segmento.

# %%
# 1. Anadir la columna 'cluster' al df original (sin escalar)
# 2. Calcular la media de cada variable por cluster
# 3. Estandarizar entre clusters y pintar un heatmap (perfil de cada segmento)

# %% [markdown]
# Leyendo el heatmap se pueden nombrar los segmentos en términos de **negocio**, por ejemplo: clientes de alto saldo y muchas compras (VIP), clientes que tiran de adelantos de efectivo (riesgo), clientes poco activos, etc. Ese nombre y la estrategia asociada es justo el entregable que pide el caso.

# %% [markdown]
# ## Detección de anomalías (Isolation Forest)
#
# Identificamos los clientes con comportamiento más atípico (posible fraude, errores de datos o clientes premium fuera de norma).

# %%
# IsolationForest sobre X, marcar los atipicos (-1) y pintarlos sobre PCA(2)

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
