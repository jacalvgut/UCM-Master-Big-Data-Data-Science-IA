# Análisis y Predicción de Series Temporales

## Descripción del Ejercicio

Este proyecto contiene un análisis completo de series temporales aplicado al **Índice de Precios de Consumo (IPC) de Hostelería y Turismo en Canarias**. El ejercicio consiste en realizar un análisis exhaustivo de una serie temporal con estacionalidad, utilizando diversos métodos de predicción y comparando sus resultados.

### Objetivos del Análisis

El trabajo incluye los siguientes apartados:

1. **Introducción**: Presentación de la serie a analizar (0.5 puntos)
2. **Representación gráfica y descomposición estacional**: Análisis visual y descomposición de la serie si presenta comportamiento estacional (1.5 puntos)
3. **División Train/Test**: Reserva de los últimos datos observados (aproximadamente un periodo estacional o 10 observaciones) para validación de los modelos
4. **Suavizado exponencial**: Encontrar el modelo de suavizado exponencial más adecuado, mostrar tabla de parámetros estimados, representación gráfica y predicciones para el periodo TEST (2 puntos)
5. **Análisis de correlogramas y modelo ARIMA manual**: Representación de la serie y correlogramas, decisión del modelo a ajustar, verificación de residuales incorrelados (2 puntos)
6. **Modelo ARIMA automático**: Ajuste automático y comparación con el modelo manual (1 punto)
7. **Expresión algebraica**: Escribir la expresión del modelo ajustado con los parámetros estimados (1 punto)
8. **Predicciones e intervalos de confianza**: Cálculo y representación gráfica de predicciones futuras (1 punto)
9. **Comparación de métodos**: Comparación de predicciones entre suavizado exponencial y ARIMA con los valores observados del TEST. Conclusiones (1 punto)

**Total: 10 puntos**

## Obtención de los Datos

### Fuente Principal: Instituto Nacional de Estadística (INE)

Los datos utilizados en este análisis provienen del **Instituto Nacional de Estadística (INE)** de España. La serie corresponde al IPC de Hostelería y Turismo en Canarias con frecuencia mensual.

**Enlace directo al INE**: https://www.ine.es/consul/serie.do?d=true&s=IPC260788&c=2&

### Características de la Serie

- **Frecuencia**: Mensual
- **Año base**: 2021 (índice 100)
- **Período**: Desde enero de 2002 hasta datos recientes (aproximadamente 150+ observaciones)
- **Interpretación**: Los valores reflejan la evolución de los precios en comparación con el nivel medio de 2021. Por ejemplo, un índice de 123,47 indica un aumento del 23,47% respecto a 2021.

### Cómo Obtener los Datos

#### Opción 1: Descarga Directa desde el INE (Recomendado)

1. **Acceso al portal del INE**:
   - Visita: https://www.ine.es/
   - Navega a "Indicadores" → "Precios" → "Índice de Precios de Consumo (IPC)"
   - Busca la serie específica: "IPC Hostelería y Turismo - Canarias"

2. **Descarga de datos**:
   - Utiliza el enlace directo proporcionado arriba
   - Selecciona el rango de fechas deseado (desde 2002 hasta la fecha más reciente disponible)
   - Descarga en formato Excel (.xlsx) o CSV

3. **Procesamiento inicial**:
   - Los datos suelen venir con filas de encabezado; es necesario saltar las primeras filas (normalmente 6 filas) al leer el archivo
   - El formato de fecha suele ser "YYYYM" (ej: "2022M01") que requiere conversión a formato fecha estándar
   - Seleccionar las columnas relevantes: PERIODO y VALOR

#### Opción 2: Plataformas de Datos Públicos

Si prefieres obtener datos ya procesados o en formatos más amigables, puedes buscar en:

**Kaggle**:
- Busca términos como: "IPC Spain", "Spanish CPI", "Canarias tourism prices"
- Filtra por datasets relacionados con economía española o turismo
- Verifica que los datos incluyan la serie mensual desde 2002

**Otros repositorios de datos**:
- **Eurostat**: https://ec.europa.eu/eurostat (datos europeos de IPC)
- **Banco de España**: https://www.bde.es/ (estadísticas económicas)
- **Repositorios académicos**: Busca en repositorios universitarios o de investigación

#### Búsqueda Recomendada en Kaggle

Para encontrar datasets similares en Kaggle, utiliza estas estrategias:

1. **Términos de búsqueda**:
   - "Spain CPI"
   - "Spanish inflation data"
   - "Canarias economic indicators"
   - "Tourism price index Spain"

2. **Filtros a aplicar**:
   - Tipo: Time Series
   - Formato: CSV o Excel
   - Idioma: Español o Inglés
   - Actualización: Datos recientes (últimos 2 años)

3. **Verificación de calidad**:
   - Comprueba que el dataset tenga al menos 150 observaciones mensuales
   - Verifica que incluya datos desde 2002 o anterior
   - Asegúrate de que la frecuencia sea mensual
   - Confirma que los valores correspondan al IPC base 2021

### Estructura de Datos Esperada

Una vez obtenidos los datos, deberías tener un archivo con la siguiente estructura:

| PERIODO | VALOR |
|---------|-------|
| 2002-01-01 | 69.466 |
| 2002-02-01 | 69.724 |
| 2002-03-01 | 70.745 |
| ... | ... |

**Nota importante**: Si los datos obtenidos tienen un formato diferente, será necesario realizar transformaciones para adaptarlos a esta estructura antes de comenzar el análisis.

## Guía para la Lectura e Interpretación de la Solución

El documento `turismo_canarias_jacobo_alvarez_gutierrez.pdf` contiene la solución completa del ejercicio. A continuación se proporciona una guía para su correcta lectura e interpretación:

### Estructura del Documento

La solución sigue el orden de los apartados del enunciado, por lo que es recomendable leerla de forma secuencial.

### Aspectos Clave por Apartado

#### 1. Introducción
- Se presenta el contexto del IPC y su relevancia para la economía canaria
- Se explica el año base y la interpretación de los valores del índice
- Se justifica la elección de esta serie por su marcada estacionalidad

#### 2. Representación Gráfica y Descomposición Estacional
- **Gráfico principal**: Muestra la evolución temporal del IPC con tendencia al alza
- **Gráfico por años**: Visualización de la estacionalidad superponiendo los valores mensuales de cada año
- **Descomposición**: Se realiza tanto con modelo aditivo como multiplicativo
  - El modelo multiplicativo suele ser más adecuado cuando los valores nunca son negativos o nulos
  - La descomposición muestra: tendencia, componente estacional y residuos

#### 3. División Train/Test
- Se reservan los últimos datos (aproximadamente 25 observaciones o 2 años) para validación
- El conjunto TRAIN se usa para ajustar los modelos
- El conjunto TEST se usa para evaluar la calidad de las predicciones

#### 4. Suavizado Exponencial (Holt-Winters)
- Se prueban diferentes variantes: Simple, Holt, Holt-Winters aditivo y multiplicativo
- Se selecciona el modelo con mejor ajuste (normalmente Holt-Winters multiplicativo para series estacionales)
- **Tabla de parámetros**: Muestra los valores estimados de α (suavizado nivel), β (tendencia), γ (estacionalidad)
- **Gráfico**: Compara la serie observada con la suavizada y las predicciones
- **Tabla de predicciones**: Valores puntuales para el periodo TEST

#### 5. Análisis ARIMA Manual
- **Correlogramas (ACF y PACF)**: 
  - ACF: Identifica el orden de diferenciación y componentes estacionales
  - PACF: Ayuda a determinar el orden AR
- **Decisión del modelo**: Basada en los patrones de los correlogramas
- **Verificación de residuales**: 
  - Los residuales deben estar incorrelados (ruido blanco)
  - Se verifica mediante test de Ljung-Box y análisis de correlogramas de residuales
- **Tabla de parámetros**: Coeficientes estimados del modelo ARIMA

#### 6. Modelo ARIMA Automático
- Se utiliza la librería `pmdarima` (auto_arima) para selección automática
- Comparación con el modelo manual mediante:
  - Criterios de información (AIC, BIC)
  - Análisis visual de ajuste
  - Calidad de las predicciones

#### 7. Expresión Algebraica
- Se presenta la ecuación del modelo con los valores numéricos de los parámetros
- Incluye las transformaciones (diferenciaciones) aplicadas
- Formato: ARIMA(p,d,q)(P,D,Q)s donde s es el período estacional (12 para datos mensuales)

#### 8. Predicciones e Intervalos de Confianza
- Se generan predicciones para períodos futuros (normalmente 25 meses o 2 años)
- Se calculan intervalos de confianza (típicamente al 95%)
- **Gráfico**: Muestra la serie histórica, predicciones y bandas de confianza

#### 9. Comparación y Conclusiones
- **Gráfico comparativo**: Superpone las predicciones de ambos métodos con los valores reales del TEST
- **Análisis de errores**: Comparación de errores absolutos o cuadráticos medios (MAE, RMSE)
- **Conclusión**: Se identifica el mejor modelo basándose en:
  - Proximidad a los valores reales
  - Comportamiento en diferentes períodos
  - Estabilidad de las predicciones

### Interpretación de Resultados

#### Métricas de Calidad
- **AIC/BIC**: Valores menores indican mejor ajuste (considerando la complejidad del modelo)
- **Error de predicción**: Menor error indica mejor capacidad predictiva
- **Intervalos de confianza**: Más estrechos indican mayor certeza en las predicciones

#### Selección del Mejor Modelo
El mejor modelo suele ser aquel que:
- Presenta residuales incorrelados (ruido blanco)
- Tiene menor error en el conjunto TEST
- Mantiene predicciones estables y coherentes con la tendencia histórica
- Captura adecuadamente la estacionalidad

### Notas Técnicas

- **Librerías utilizadas**: pandas, numpy, matplotlib, seaborn, statsmodels, pmdarima
- **Procesamiento**: Los datos requieren limpieza y transformación de fechas
- **Advertencias**: Se suprimen warnings comunes de openpyxl y statsmodels para limpieza del output

### Recomendaciones para el Estudio

1. **Lectura secuencial**: Sigue el orden de los apartados para entender la progresión del análisis
2. **Revisa los gráficos**: Los gráficos son fundamentales para entender el comportamiento de la serie
3. **Presta atención a las tablas**: Contienen los valores numéricos clave de los modelos
4. **Analiza las conclusiones**: La comparación final resume los hallazgos más importantes
5. **Replica el código**: Si tienes acceso al código Python, intenta ejecutarlo para entender mejor cada paso

## Estructura del Repositorio

```
MOD07_03/
├── README.md                                    # Este archivo
├── turismo_canarias_jacobo_alvarez_gutierrez.pdf # Solución completa del ejercicio
└── Ejercicio de Evaluación Series.pdf          # Enunciado del ejercicio (referencia)
```

**Nota**: Los datos deben obtenerse siguiendo las instrucciones de la sección "Obtención de los Datos" de este README.

## Requisitos Técnicos

Para replicar el análisis se requiere:

- Python 3.x
- Librerías principales:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - statsmodels
  - pmdarima
  - openpyxl (para leer archivos Excel)

## Autor

Jacobo Álvarez Gutiérrez

---

**Nota**: Este README proporciona una guía completa para entender tanto el ejercicio como su solución. Se recomienda leer primero el documento de solución completo para obtener todos los detalles técnicos y visualizaciones.

