# Análisis de Componentes Principales (PCA) y Clustering - Pingüinos

## Descripción de la Tarea

Este proyecto tiene por objetivo aplicar técnicas de análisis de datos con fines predictivos, concretamente:

- **Análisis de Componentes Principales (PCA)**: Técnica de reducción de dimensionalidad que permite identificar las variables más importantes y reducir el número de variables explicativas manteniendo la mayor parte de la información.
- **Análisis Clustering**: Técnica de agrupación no supervisada que permite identificar patrones y agrupar observaciones similares en clústeres.

## Base de Datos

El conjunto de datos utilizado es **`penguins.xlsx`**, que contiene información sobre pingüinos con las siguientes variables:

- **Variables categóricas**:
  - `species`: Especie del pingüino (Adelie, Chinstrap, Gentoo)
  - `island`: Isla de procedencia
  - `sex`: Género del pingüino

- **Variables numéricas**:
  - `bill_length_mm`: Longitud del pico (mm)
  - `bill_depth_mm`: Profundidad del pico (mm)
  - `flipper_length_mm`: Longitud de la aleta (mm)
  - `body_mass_g`: Masa corporal (g)

## Protocolo de Análisis

### 1. Preparación del Entorno y Depuración de Datos

1. **Importación de librerías**: Se importan las librerías necesarias:
   - `pandas` y `numpy` para manipulación de datos
   - `seaborn` y `matplotlib` para visualización
   - `sklearn` para PCA, escalado y clustering
   - `scipy` para análisis jerárquico

2. **Lectura de datos**: Se carga el archivo `penguins.xlsx` usando `pd.read_excel()`

3. **Separación de variables**: Se separan las variables numéricas de las categóricas para el análisis

4. **Estadísticos descriptivos**: Se calculan estadísticos descriptivos (mínimo, percentiles, mediana, media, máximo, desviación estándar, varianza) y se verifica la presencia de valores perdidos

5. **Análisis de correlación**: 
   - Se calcula la matriz de covarianzas
   - Se calcula la matriz de correlaciones
   - Se visualiza la matriz de correlaciones mediante un mapa de calor

### 2. Análisis de Componentes Principales (PCA)

1. **Normalización de datos**: Se estandarizan las variables numéricas usando `StandardScaler()` para que todas tengan media 0 y desviación estándar 1

2. **Aplicación de PCA**: 
   - Se aplica PCA inicialmente a todas las variables (4 componentes)
   - Se obtienen los autovalores y autovectores
   - Se analiza la varianza explicada por cada componente

3. **Selección del número de componentes**:
   - Se analiza la varianza explicada y acumulada
   - Se visualiza mediante gráfico de "codo"
   - Se seleccionan las 2 primeras componentes principales

4. **Análisis de relaciones**:
   - Se calculan las correlaciones entre variables originales y componentes principales
   - Se calculan los cosenos cuadrados (proporción de variabilidad explicada)
   - Se visualizan mediante mapas de calor y gráficos de barras
   - Se representan gráficamente las contribuciones mediante vectores

5. **Representación de registros**:
   - Se visualizan los registros en el espacio de las componentes principales
   - Se incluyen los centroides de las especies
   - Se representan las contribuciones de las variables mediante vectores

### 3. Análisis Clustering

#### 3.1. Clustering Jerárquico

1. **Matriz de distancias**: Se calcula la matriz de distancias euclídeas entre todos los pares de observaciones usando los datos normalizados

2. **Visualización**: 
   - Se genera un mapa de calor con clustering jerárquico
   - Se construye un dendrograma usando el método de Ward

3. **Asignación de clústeres**: Se asignan las observaciones a clústeres basándose en el dendrograma

4. **Visualización en espacio PCA**: Se representan los clústeres en el espacio de las componentes principales

#### 3.2. Clustering No Jerárquico (K-means)

1. **Selección del número óptimo de clústeres**:
   - **Método del codo**: Se calcula el WCSS (Within-Cluster Sum of Squares) para diferentes valores de K (1 a 10) y se identifica el punto de inflexión
   - **Método de la silueta**: Se calcula el coeficiente de silueta para diferentes valores de K (2 a 10) y se selecciona el valor que maximiza este coeficiente

2. **Aplicación de K-means**: Se aplica el algoritmo K-means con el número óptimo de clústeres determinado

3. **Evaluación de la calidad**:
   - Se calculan los valores de silueta para cada observación
   - Se visualiza mediante gráfico de barras de silueta por clúster
   - Se interpreta la calidad de la agrupación

4. **Análisis de resultados**:
   - Se visualizan los clústeres en el espacio PCA junto con los centroides de especies e islas
   - Se calculan los centroides de cada clúster en las variables originales
   - Se interpretan las diferencias entre clústeres

## Resultados Principales

- **PCA**: Las dos primeras componentes principales explican la mayor parte de la varianza. La primera componente está relacionada principalmente con la masa corporal y la longitud de la aleta, mientras que la segunda componente está relacionada con las características del pico.

- **Clustering**: El análisis sugiere que 3 clústeres es el número óptimo, que muestra una buena correspondencia con las especies de pingüinos, especialmente con la especie Gentoo que se diferencia claramente de las otras dos especies.

## Requisitos

Las librerías necesarias se encuentran en el notebook y incluyen:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- scipy

## Archivos

- `jacobo_alvarez_gutierrez_penguins.ipynb`: Notebook con el análisis completo
- `penguins.xlsx`: Archivo de datos (debe estar en el mismo directorio que el notebook)

