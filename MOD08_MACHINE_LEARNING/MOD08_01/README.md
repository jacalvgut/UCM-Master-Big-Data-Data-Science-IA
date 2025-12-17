# Trabajo de Clasificación Binaria - Machine Learning

## Descripción del Proyecto

Este proyecto desarrolla un modelo de clasificación binaria para predecir si un vehículo debe ser repintado de color blanco o no, basándose en características técnicas y comerciales de automóviles usados. El problema se aborda desde la perspectiva de una empresa dedicada a la venta de coches usados que necesita determinar el color óptimo (blanco o negro) para repintar vehículos que llegan en condiciones deficientes.

### Contexto del Problema

La empresa ha decidido limitarse a los colores blanco y negro por ser los más comunes en el mercado. Para tomar esta decisión, se desarrolla un modelo predictivo que, basándose en las características de los vehículos en el mercado de segunda mano, determine si originalmente eran blancos o negros.

### Variable Objetivo

La variable objetivo se construye a partir del campo original 'Color', limitado exclusivamente a las categorías blanco y negro:
- **1**: Blanco (White)
- **0**: Negro (Black)

### Variables Predictoras

La base de datos incluye las siguientes variables independientes:
- Precio de venta (Price)
- Cantidad de Impuestos a pagar (Levy)
- Fabricante (Manufacturer)
- Año de fabricación (Prod. year)
- Categoría (Category)
- Interior de cuero (Leather interior)
- Tipo de combustible (Fuel type)
- Volumen del motor (Engine volume)
- Kilometraje (Mileage)
- Cilindros (Cylinders)
- Tipo de caja de cambios (Gear box type)
- Ruedas motrices (Drive wheels)
- Lugar del volante (Wheel)
- Número de Airbags (Airbags)

### Base de Datos

Este proyecto utiliza una base de datos derivada del dataset **"Car Price Prediction Challenge"** disponible en Kaggle. La base de datos original contiene información sobre vehículos usados con múltiples características técnicas y comerciales.

**Fuente original**: [Car Price Prediction Challenge - Kaggle](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge)

#### Modificaciones Realizadas a la Base Original

Para adaptar la base de datos al problema de clasificación binaria, se realizaron las siguientes modificaciones:

1. **Filtrado por color**: Se eliminaron todas las filas con colores diferentes a blanco o negro, dejando únicamente vehículos de estos dos colores.

2. **Eliminación de variables con muchas categorías**: Se eliminaron variables con alta cardinalidad que dificultan el modelado:
   - Identificador del vehículo
   - Modelo del vehículo
   - Número de puertas

3. **Filtrado de categorías poco representadas**: Se eliminaron categorías con menos de 625 observaciones en las siguientes variables categóricas:
   - Fabricante (Manufacturer)
   - Categoría (Category)
   - Tipo de combustible (Fuel type)
   - Tipo de caja de cambios (Gear box type)
   - Ruedas motrices (Drive wheels)

4. **Construcción de la variable objetivo**: La variable 'Color' se transformó en una variable binaria donde:
   - **1** = Blanco (White)
   - **0** = Negro (Black)

**Nota**: El código espera un archivo Excel (`datos_tarea25.xlsx`) con estas modificaciones ya aplicadas. Si deseas trabajar con la base de datos original de Kaggle, deberás aplicar estas transformaciones previamente.

## Estructura del Código

El código está organizado en tres apartados principales:

### 1. Análisis y Depuración de la Base de Datos
- **Consideraciones generales**: Corrección de tipos de datos y transformaciones básicas
- **Valores atípicos**: Identificación y eliminación de outliers
- **Valores missing**: Análisis y estrategias de imputación
- **Selección de variables**: Análisis de correlaciones (Pearson y V de Cramér)
- **Normalización y dummificación**: Preparación de datos para modelos de ML

### 2. Modelo con Máquina de Vectores Soporte (SVM)
- Búsqueda de hiperparámetros para kernels lineal y RBF
- Comparación gráfica de resultados (accuracy y AUC)
- Aplicación de bagging al mejor modelo encontrado
- Evaluación mediante validación cruzada

### 3. Modelo con Técnica Stacking
- Desarrollo de modelos base:
  - Regresión Logística
  - Random Forest
  - XGBoost
  - SVM con kernel RBF
- Comparación de modelos base
- Ensamblado stacking con regresión logística como meta-modelo

## Requisitos Previos

### Dependencias de Python

El código requiere las siguientes librerías:

```python
numpy
pandas
seaborn
scipy
matplotlib
joblib
scikit-learn
xgboost
openpyxl  # Para leer archivos Excel
```

### Instalación de Dependencias

```bash
pip install numpy pandas seaborn scipy matplotlib joblib scikit-learn xgboost openpyxl
```

### Obtención de la Base de Datos

**IMPORTANTE**: Este repositorio no incluye el archivo de datos. Debes obtenerlo siguiendo estos pasos:

1. **Opción 1 - Base de datos modificada** (recomendada):
   - Si tienes acceso al archivo `datos_tarea25.xlsx` con las modificaciones descritas anteriormente, colócalo en tu directorio de trabajo.

2. **Opción 2 - Base de datos original de Kaggle**:
   - Descarga el dataset original desde: [Car Price Prediction Challenge](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge)
   - Aplica las modificaciones descritas en la sección "Base de Datos" (filtrado por color, eliminación de variables, etc.)
   - Guarda el resultado como `datos_tarea25.xlsx` en formato Excel

**Formato esperado**: El código espera un archivo Excel (`.xlsx`) con las columnas descritas en la sección "Variables Predictoras" y la variable objetivo 'Color' codificada como 1 (blanco) o 0 (negro).

### Configuración de Rutas

El código contiene rutas hardcodeadas que deben ajustarse según tu sistema:
- Línea 33: `os.chdir('C:/Users/jacob/Desktop/master_ucm/mod8_machine_learning/tema_1/Tarea')`
- Línea 271: Misma ruta
- Línea 550: Misma ruta

**Recomendación**: Modifica estas rutas para que apunten al directorio donde coloques el archivo `datos_tarea25.xlsx`.

## Instrucciones de Ejecución

### Paso 1: Preparación del Entorno

1. **Obtén la base de datos**: Descarga o prepara el archivo `datos_tarea25.xlsx` siguiendo las instrucciones de la sección "Obtención de la Base de Datos"
2. **Coloca el archivo**: Coloca `datos_tarea25.xlsx` en un directorio accesible
3. **Configura las rutas**: Modifica las rutas en el código Python (líneas 33, 271, 550) para que apunten a tu directorio de trabajo
4. **Instala dependencias**: Asegúrate de tener todas las dependencias instaladas (ver sección "Requisitos Previos")

### Paso 2: Ejecución del Código

El código está diseñado para ejecutarse secuencialmente. Ejecuta el script completo:

```bash
python tarea_final_machinelearning_jacobo_alvarez_gutierrez.py
```

### Paso 3: Archivos Generados

Durante la ejecución, el código generará los siguientes archivos intermedios:

- `datos_null_transf.xlsx`: Base de datos con imputación nula, normalizada y dummificada
- `datos_central_transf.xlsx`: Base de datos con imputación central, normalizada y dummificada
- `null_scaler.pkl`: Preprocesador guardado para imputación nula
- `central_scaler.pkl`: Preprocesador guardado para imputación central

**Nota**: Estos archivos son necesarios para la ejecución completa del código. Si los eliminas, deberás ejecutar nuevamente el Apartado 1.

## Interpretación de Resultados

### Apartado 1: Depuración de Datos

#### Valores Atípicos Eliminados
- **Engine volume**: Se eliminaron valores 0, 0.1 y 0.6 (físicamente imposibles)
- **Cylinders**: Se eliminaron valores 1, 2 y 7 (poco comunes o erróneos)
- **Mileage**: Se eliminó el valor 1111111111 (claramente erróneo)
- **Price**: Se eliminaron valores fuera del rango intercuartílico (método IQR en escala logarítmica)

#### Estrategias de Imputación
El código implementa dos estrategias para manejar valores missing en 'Levy':
1. **Imputación nula**: Asigna 0 a valores faltantes (representa exención de impuestos)
2. **Imputación central**: Asigna la mediana según el número de cilindros

Ambas estrategias se comparan mediante análisis de correlación.

### Apartado 2: Modelo SVM

#### Búsqueda de Hiperparámetros

**Kernel Lineal**:
- Se busca el parámetro `C` óptimo mediante GridSearchCV
- Se generan gráficos comparando accuracy y AUC para diferentes valores de C
- Se comparan resultados entre ambas estrategias de imputación

**Kernel RBF**:
- Se busca la combinación óptima de `C` y `gamma`
- Se generan heatmaps comparativos para accuracy y AUC
- Se visualizan los resultados para ambas estrategias de imputación

#### Resultados Esperados

Según la memoria del proyecto:
- **Mejor kernel**: RBF con C=10 y gamma=100
- **Accuracy**: ~0.78
- **AUC**: ~0.88

#### Bagging

Se aplica bagging al mejor modelo SVM encontrado:
- **Número de estimadores**: 10
- **Muestra**: 80% con bootstrap
- **Resultado**: Mejora marginal o similar al modelo base

### Apartado 3: Modelo Stacking

#### Modelos Base

Cada modelo base se optimiza mediante GridSearchCV:

1. **Regresión Logística**:
   - Parámetros optimizados: C, solver, penalty, class_weight
   - Rendimiento: Moderado

2. **Random Forest**:
   - Parámetros optimizados: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
   - Rendimiento: Alto

3. **XGBoost**:
   - Parámetros optimizados: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_lambda
   - Rendimiento: Alto

4. **SVM RBF**:
   - Usa los mismos parámetros del mejor modelo del Apartado 2
   - Rendimiento: Alto

#### Comparación de Modelos Base

El código genera gráficos de boxplot mostrando:
- Distribución de accuracy en validación cruzada (5 folds)
- Distribución de AUC en validación cruzada (5 folds)

**Interpretación**: Los modelos con mejor rendimiento individual (Random Forest y XGBoost) suelen ser los que más contribuyen al stacking.

#### Modelo Stacking Final

- **Meta-modelo**: Regresión Logística
- **Método de stacking**: predict_proba
- **Validación cruzada**: 5 folds estratificados

**Resultados Esperados**:
- **Accuracy**: ~0.7873
- **AUC**: ~0.8836
- **Mejora**: Ligera mejora respecto a los modelos base individuales

### Métricas de Evaluación

#### Accuracy
- **Definición**: Proporción de predicciones correctas sobre el total
- **Interpretación**: Valores cercanos a 1 indican mejor rendimiento
- **Rango**: [0, 1]

#### AUC (Area Under the ROC Curve)
- **Definición**: Área bajo la curva ROC
- **Interpretación**: 
  - 0.5: Clasificador aleatorio
  - 0.7-0.8: Aceptable
  - 0.8-0.9: Bueno
  - >0.9: Excelente
- **Rango**: [0, 1]

#### Matriz de Confusión
El código genera matrices de confusión para visualizar:
- **Verdaderos Positivos (TP)**: Correctamente predichos como blanco
- **Verdaderos Negativos (TN)**: Correctamente predichos como negro
- **Falsos Positivos (FP)**: Incorrectamente predichos como blanco
- **Falsos Negativos (FN)**: Incorrectamente predichos como negro

**Interpretación**: Una matriz equilibrada indica que el modelo no tiene sesgo hacia ninguna clase.

## Notas Importantes

1. **Tiempo de Ejecución**: El código puede tardar varios minutos en ejecutarse completamente debido a:
   - Múltiples GridSearchCV con validación cruzada
   - Procesamiento de datos
   - Generación de gráficos

2. **Memoria**: El procesamiento de datos y modelos puede requerir memoria considerable, especialmente durante el stacking.

3. **Reproducibilidad**: El código usa `random_state=123` en múltiples lugares para garantizar resultados reproducibles.

4. **Gráficos**: El código genera múltiples gráficos que se mostrarán durante la ejecución. Asegúrate de tener una sesión interactiva o ajusta el código para guardar los gráficos en archivos.

## Estructura de Archivos Esperada

```
directorio_trabajo/
├── datos_tarea25.xlsx                    # Archivo de datos (debe obtenerse externamente)
├── tarea_final_machinelearning_jacobo_alvarez_gutierrez.py  # Código principal
├── datos_null_transf.xlsx                # Generado en Apartado 1 (se crea al ejecutar)
├── datos_central_transf.xlsx             # Generado en Apartado 1 (se crea al ejecutar)
├── null_scaler.pkl                       # Generado en Apartado 1 (se crea al ejecutar)
├── central_scaler.pkl                    # Generado en Apartado 1 (se crea al ejecutar)
└── README.md                             # Este archivo
```

**Nota**: El archivo `datos_tarea25.xlsx` no está incluido en este repositorio. Debes obtenerlo siguiendo las instrucciones de la sección "Obtención de la Base de Datos".

## Referencias

### Base de Datos

- **Dataset original**: [Car Price Prediction Challenge - Kaggle](https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge)
  - Contiene información sobre vehículos usados con múltiples características técnicas y comerciales
  - Incluye variables como precio, características del motor, especificaciones técnicas, etc.
  - Requiere modificaciones (filtrado y transformaciones) para adaptarse al problema de clasificación binaria

### Documentación Técnica

- Documentación scikit-learn: https://scikit-learn.org/stable/
- Documentación XGBoost: https://xgboost.readthedocs.io/
- Documentación pandas: https://pandas.pydata.org/docs/

## Autor

Jacobo Álvarez Gutiérrez

---

**Nota Final**: Este README está diseñado para facilitar la comprensión y ejecución del código. Para una explicación detallada de la metodología y resultados, consulta la memoria del proyecto (`memoria_jacobo_alvarez_gutierrez.pdf`).

