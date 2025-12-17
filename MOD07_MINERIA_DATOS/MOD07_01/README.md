# Análisis de Elecciones Españolas: Modelización Predictiva con Regresión Lineal y Logística

## Resumen del Proyecto

Este proyecto tiene como objetivo desarrollar modelos predictivos para analizar los patrones de abstención electoral en los municipios españoles a partir de variables demográficas y socioeconómicas. El estudio se centra en dos variables objetivo relacionadas con la abstención electoral:

- **AbstentionPtge**: Variable continua que representa el porcentaje de abstención electoral.
- **AbstencionAlta**: Variable dicotómica que toma el valor 1 si el porcentaje de abstención es superior al 30%, y 0 en otro caso.

Se construyen dos tipos de modelos:
1. **Modelo de Regresión Lineal**: Para predecir el porcentaje de abstención (variable continua).
2. **Modelo de Regresión Logística**: Para predecir si la abstención es alta o no (variable binaria).

El conjunto de datos contiene información demográfica y socioeconómica de distintos municipios de España junto con los resultados de las últimas elecciones, incluyendo variables como población, distribución por edades, actividad económica principal, densidad poblacional, porcentaje de extranjeros, y otras características relevantes de cada municipio.

## Estructura del Repositorio

- `tarea_final_jacobo_alvarez_gutierrez.py`: Script principal en Python que contiene toda la implementación de la solución, desde la depuración de datos hasta la construcción y evaluación de los modelos predictivos.
- `tarea_final_jacobo_alvarez_gutierrez.pdf`: Documento detallado que presenta el análisis completo, incluyendo justificaciones metodológicas, códigos, salidas y comentarios de los resultados obtenidos.

## Guía de Lectura e Interpretación de la Solución

El código Python está organizado en secciones claramente definidas que siguen un flujo lógico de análisis de datos. A continuación se explica la estructura y cómo interpretar cada parte:

### 1. Depuración de Datos (Líneas 1-205)

Esta primera sección se encarga de preparar los datos para el análisis:

- **Importación y configuración inicial** (líneas 11-35): Se importan las librerías necesarias (pandas, numpy, matplotlib, seaborn, sklearn) y funciones personalizadas del módulo `FuncionesMineria`. El dataset se carga desde un archivo Excel.

- **Selección de variables objetivo** (líneas 37-42): Se eliminan las variables objetivo no utilizadas (Izda_Pct, Dcha_Pct, Otros_Pct, Izquierda, Derecha), manteniendo solo AbstentionPtge y AbstencionAlta.

- **Asignación correcta de tipos de variables** (líneas 44-62): Se corrige la asignación errónea de variables categóricas que fueron importadas como numéricas (AbstencionAlta y CodigoProvincia) y se clasifican todas las variables en numéricas y categóricas.

- **Análisis descriptivo y detección de errores** (líneas 64-109): 
  - Se analizan las variables categóricas para detectar categorías poco representadas o errores de codificación.
  - Se calculan estadísticos descriptivos para variables numéricas (media, desviación estándar, asimetría, curtosis, rango).
  - Se identifican y corrigen errores como valores fuera de rango en porcentajes (valores > 100 o < 0) y valores codificados erróneamente (99999 para missing, '?' para valores perdidos).

- **Tratamiento de datos atípicos** (líneas 128-137): Se detectan valores atípicos mediante métodos estadísticos y se convierten a valores missing, ya que la proporción de datos atípicos es pequeña en todas las variables.

- **Análisis y tratamiento de valores perdidos** (líneas 139-198):
  - Se analiza la proporción de valores perdidos por variable y por observación.
  - Se crea una variable auxiliar `prop_missings` que representa la proporción de valores perdidos por observación.
  - Se imputan los valores missing: mediante imputación aleatoria para variables categóricas y cuantitativas.

- **Guardado de datos depurados** (líneas 200-205): Los datos depurados se guardan en un archivo pickle para su posterior uso.

### 2. Análisis de Relaciones entre Variables (Líneas 207-243)

Esta sección explora las relaciones entre las variables explicativas y las variables objetivo:

- **Correlación de Pearson** (líneas 220-231): Se calcula y visualiza una matriz de correlación entre todas las variables numéricas continuas, lo que permite identificar relaciones lineales fuertes.

- **Estadístico V de Cramer** (líneas 233-236): Se analiza la asociación entre variables categóricas y las variables objetivo mediante el estadístico V de Cramer, que mide la fuerza de la asociación entre variables categóricas.

- **Selección inicial de variables** (líneas 238-243): Basándose en los análisis anteriores, se realiza una primera selección de variables más relevantes:
  - Variables continuas: `totalEmpresas`, `TotalCensus`, `PersonasInmueble`, `Age_0-4_Ptge`, `Age_under19_Ptge`, `Age_over65_pct`
  - Variables categóricas: `CCAA`, `CodigoProvincia`, `ActividadPpal`

### 3. Modelos de Regresión Lineal (Líneas 245-544)

Se construyen múltiples modelos de regresión lineal para predecir el porcentaje de abstención:

- **Partición train/test** (líneas 249-254): Se divide el conjunto de datos en entrenamiento (80%) y prueba (20%) con una semilla aleatoria fija para reproducibilidad.

- **Métodos de selección clásica de variables** (líneas 256-316):
  - Se aplican tres métodos: Forward, Backward y Stepwise.
  - Cada método se prueba con dos criterios de selección: AIC y BIC.
  - Para cada modelo se muestra el summary, se calcula R² en entrenamiento y prueba, y se evalúa el rendimiento.

- **Modelos con interacciones** (líneas 318-342): Se generan modelos adicionales incluyendo interacciones entre variables continuas, aplicando Stepwise con AIC y BIC.

- **Comparación de modelos** (líneas 344-357): Se verifica si algunos modelos son idénticos mediante la función `comparar_modelos`.

- **Validación cruzada** (líneas 359-422): Se realiza validación cruzada con 20 repeticiones y 5 folds para comparar los modelos candidatos. Los resultados se visualizan mediante boxplots y se calculan medias y desviaciones estándar de R².

- **Selección aleatoria de variables** (líneas 428-534): 
  - Se realiza selección aleatoria mediante 30 iteraciones, en cada una se divide el conjunto de entrenamiento y se aplica Stepwise con BIC.
  - Se identifican los modelos más frecuentes entre las iteraciones.
  - Se comparan estos modelos con el mejor modelo de los métodos clásicos mediante validación cruzada.

- **Modelo ganador** (líneas 536-544): Se identifica el modelo ganador (modeloStepBIC en este caso) y se evalúan sus características finales, incluyendo el tamaño del efecto de las variables y el R² en entrenamiento y prueba.

### 4. Modelos de Regresión Logística (Líneas 546-910)

Se construyen modelos de regresión logística para predecir si la abstención es alta (binaria):

- **Partición train/test** (líneas 554-558): Similar a la regresión lineal, pero utilizando la variable objetivo binaria (AbstencionAlta).

- **Métodos de selección clásica** (líneas 560-622): Se aplican los mismos métodos que en regresión lineal (Forward, Backward, Stepwise con AIC y BIC), pero utilizando funciones GLM (`glm_forward`, `glm_backward`, `glm_stepwise`). Las métricas utilizadas son pseudo R² y el área bajo la curva ROC (AUC).

- **Modelos con interacciones** (líneas 639-667): Se añaden interacciones entre variables continuas (con consideraciones sobre multicolinealidad).

- **Validación cruzada** (líneas 670-725): Similar a la regresión lineal, pero evaluando el AUC en lugar de R².

- **Selección aleatoria de variables** (líneas 727-820): Se aplica el mismo proceso que en regresión lineal, identificando los modelos más frecuentes mediante 30 iteraciones.

- **Punto de corte óptimo** (líneas 824-876): Se determina el punto de corte óptimo para clasificar las predicciones como "abstención alta" o "no alta". Se evalúan dos criterios:
  - Índice de Youden (sensibilidad + especificidad - 1)
  - Accuracy (precisión)
  Se generan gráficos para visualizar cómo varían estas métricas según el punto de corte y se selecciona el óptimo (0.62 en este caso, basado en Accuracy).

- **Modelo ganador** (líneas 879-910): Se evalúa el modelo ganador (modeloBackAIC_glm) comparando las métricas entre entrenamiento y prueba para verificar la estabilidad del modelo. Se calculan pseudo R², AUC, y métricas de clasificación (sensibilidad, especificidad, valor predictivo positivo y negativo).

### Conceptos Clave para la Interpretación

- **R² (R-cuadrado)**: En regresión lineal, mide la proporción de varianza explicada por el modelo. Valores más altos (cercanos a 1) indican mejor ajuste.

- **Pseudo R²**: En regresión logística, es una medida análoga al R² que indica la bondad de ajuste del modelo.

- **AUC (Área Bajo la Curva ROC)**: En clasificación binaria, mide la capacidad del modelo para distinguir entre clases. Valores cercanos a 1 indican mejor rendimiento.

- **AIC y BIC**: Criterios de información que penalizan la complejidad del modelo. BIC suele preferir modelos más simples que AIC.

- **Validación cruzada**: Técnica para evaluar la generalización del modelo, evitando sobreajuste.

- **Punto de corte (threshold)**: En clasificación binaria, es el umbral de probabilidad utilizado para clasificar una observación en una u otra categoría.

## Obtención del Conjunto de Datos

El archivo `DatosEleccionesEspaña.xlsx` utilizado en este proyecto no está incluido en el repositorio debido a restricciones de tamaño y posibles restricciones de uso. Este conjunto de datos contiene información demográfica y electoral de municipios españoles.

### Características del Dataset

El dataset original debería incluir las siguientes variables:

**Variables objetivo (se utilizan 2 de 7 disponibles):**
- `AbstentionPtge`: Porcentaje de abstención (variable continua utilizada)
- `AbstencionAlta`: Variable binaria (1 si abstención > 30%, 0 en otro caso; utilizada)
- `Izda_Pct`: Porcentaje de votos a partidos de izquierda (no utilizada)
- `Dcha_Pct`: Porcentaje de votos a partidos de derecha (no utilizada)
- `Otros_Pct`: Porcentaje de votos a otros partidos (no utilizada)
- `Izquierda`: Variable binaria para mayoría de izquierda (no utilizada)
- `Derecha`: Variable binaria para mayoría de derecha (no utilizada)

**Variables explicativas (ejemplos del código):**
- Demográficas: `TotalCensus`, `PersonasInmueble`, `Age_0-4_Ptge`, `Age_under19_Ptge`, `Age_19_65_pct`, `Age_over65_pct`, `ForeignersPtge`
- Geográficas: `CCAA`, `CodigoProvincia`, `Densidad`
- Económicas: `totalEmpresas`, `ActividadPpal`, `Explotaciones`
- Otras: `SameComAutonPtge`

### Dónde Obtener el Dataset

Para obtener un dataset similar o equivalente, puede consultar las siguientes fuentes:

1. **Kaggle** (https://www.kaggle.com): 
   - Buscar datasets relacionados con "Spanish elections", "elecciones España", "municipalidades España", o "Spanish municipalities demographic data".
   - Kaggle alberga numerosos datasets públicos sobre elecciones y datos demográficos de diferentes países.

2. **Instituto Nacional de Estadística (INE)** (https://www.ine.es):
   - Portal oficial de estadísticas de España que proporciona datos demográficos y socioeconómicos de municipios españoles.
   - Los datos electorales pueden obtenerse del Ministerio del Interior.

3. **datos.gob.es** (https://datos.gob.es):
   - Portal de datos abiertos del gobierno español que centraliza información pública disponible para reutilización.

4. **Repositorios académicos**:
   - Algunas universidades y centros de investigación comparten datasets educativos. Puede consultar repositorios de datos de universidades españolas o plataformas como Zenodo.

### Notas Importantes

- **Requisitos del código**: El script requiere un módulo personalizado llamado `FuncionesMineria` que contiene funciones auxiliares para minería de datos. Este módulo debe estar disponible en el directorio de trabajo o en el path de Python para que el código funcione correctamente.

- **Formato de datos**: El código espera un archivo Excel (`.xlsx`) con el nombre `DatosEleccionesEspaña.xlsx`. Si obtiene los datos en otro formato (CSV, por ejemplo), deberá modificar la línea 35 del script para usar `pd.read_csv()` en lugar de `pd.read_excel()`.

- **Configuración del directorio**: En la línea 22 del script se establece el directorio de trabajo. Deberá modificarlo según su configuración local o comentar esa línea si ya está ejecutando desde el directorio correcto.

- **Reproducibilidad**: El código utiliza semillas aleatorias (`random_state`) para garantizar la reproducibilidad de los resultados. Si desea variabilidad, puede cambiar o eliminar estos parámetros.

## Ejecución del Código

Para ejecutar el script completo:

```bash
python tarea_final_jacobo_alvarez_gutierrez.py
```

**Requisitos previos:**
- Python 3.x
- Librerías: pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl (para leer Excel)
- Módulo `FuncionesMineria` con las funciones auxiliares necesarias

## Contacto y Autor

Proyecto desarrollado por **Jacobo Álvarez Gutiérrez** como parte del Máster en Big Data, Data Science & Inteligencia Artificial de la Universidad Complutense de Madrid.

