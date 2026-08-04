# Datos procesados (`processed`)

Datos **listos para modelar**, resultado de las transformaciones finales aplicadas a los datos limpios:

- **Setas (Parte 1):** One-Hot Encoding (`pd.get_dummies`) sobre las variables categóricas y, opcionalmente, reducción de dimensionalidad (PCA).
- **Tarjetas (Parte 2):** escalado con `StandardScaler` (imprescindible por las diferencias de escala entre variables) y, opcionalmente, proyección PCA.

## Convención

- Carpeta **actualmente vacía**: en este taller las transformaciones se aplican **dentro de los notebooks** y no se persisten.
- Si se quisiera guardar el dataset final transformado para reutilizarlo (por ejemplo, para visualizaciones o para otros experimentos), se volcaría aquí con un nombre descriptivo (p. ej. `mushrooms_encoded.csv`, `credit_card_scaled.csv`).
- Los archivos de esta carpeta deben considerarse **derivables**: se pueden regenerar siempre a partir de `data/raw/`.
