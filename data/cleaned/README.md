# Datos limpios (`cleaned`)

Resultados intermedios de la etapa de **limpieza de datos**: nulos imputados, valores `?` tratados, columnas constantes eliminadas y atípicos gestionados.

## Convención

- Carpeta **actualmente vacía**: en este taller la limpieza se ejecuta **dentro de los notebooks** a partir de `data/raw/`.
- Si se quiere reutilizar una versión limpia (por ejemplo, para ahorrar tiempo o para asegurar reproducibilidad), los datasets ya imputados se guardarían aquí con nombres descriptivos (p. ej. `mushrooms_clean.csv`, `credit_card_clean.csv`).
- A partir de aquí, los datos aún **no** están listos para modelar: falta el escalado/encoding que ocurre en `data/processed/`.
