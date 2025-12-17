# Tarea de Deep Learning - Redes Neuronales

## Descripción General

Esta tarea está dividida en dos actividades principales que cubren aspectos fundamentales del aprendizaje profundo:

1. **Actividad 1: Redes Densas (Dense Networks)** - Regresión con datos tabulares
2. **Actividad 2: Redes Convolucionales (CNNs)** - Clasificación de imágenes

La tarea tiene una puntuación máxima de 10 puntos, distribuidos equitativamente entre ambas actividades.

---

## Actividad 1: Redes Densas (5 puntos)

### Objetivo
Predecir la calidad del vino utilizando el [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). Se trata de un problema de **regresión**, ya que la calidad puede tomar valores decimales (aunque en el dataset aparezcan como enteros).

### Dataset
- Combina datos de vinos tintos y blancos
- 11 características (features): acidez fija, acidez volátil, ácido cítrico, azúcar residual, cloruros, dióxido de azufre libre, dióxido de azufre total, densidad, pH, sulfatos y alcohol
- Variable objetivo: calidad del vino (valores enteros en el dataset, pero el modelo debe predecir valores decimales)

### Tareas Implementadas

#### Normalización (0.25 pts)
- Normalización de las features usando la capa `Normalization` de Keras
- Se adapta únicamente con los datos de entrenamiento

#### Cuestión 1 (1.5 pts)
- Creación de un modelo secuencial con **4 capas ocultas**
- Cada capa oculta tiene **más de 60 neuronas** (implementado con 128 neuronas)
- Sin técnicas de regularización
- Función de pérdida: `mse` (Mean Squared Error) - apropiada para regresión
- Métrica: `mae` (Mean Absolute Error)
- Resultado: Test Loss ≈ 0.559, Test MAE ≈ 0.536

#### Cuestión 2 (1.5 pts)
- Mismo modelo base pero con **dos técnicas de regularización**:
  - **Regularización L2** (`kernel_regularizer=l2(0.001)`) en todas las capas densas
  - **Dropout** (tasa 0.3) después de cada capa oculta
- Resultado: Test Loss ≈ 0.536, Test MAE ≈ 0.525 (mejora respecto al modelo sin regularización)

#### Cuestión 3 (0.5 pts)
- Implementación de **Early Stopping** callback
- Monitoriza `val_loss` con paciencia de 10 épocas
- Restaura los mejores pesos al finalizar el entrenamiento
- Resultado: Test Loss ≈ 0.523, Test MAE ≈ 0.529

#### Cuestiones Teóricas (1.25 pts)
- **Cuestión 4**: Análisis de funciones de activación para la capa de salida en regresión
- **Cuestión 5**: Explicación del cálculo de una neurona
- **Cuestión 6**: Identificación de funciones de activación no recomendadas en capas ocultas
- **Cuestión 7**: Técnicas efectivas para combatir el overfitting
- **Cuestión 8**: Configuración de capa de salida y función de pérdida para clasificación multiclase

---

## Actividad 2: Redes Convolucionales (5 puntos)

### Objetivo
Clasificar imágenes del dataset [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) utilizando redes convolucionales.

### Dataset
- **CIFAR-10**: 60,000 imágenes de 32x32 píxeles a color (RGB)
- **10 clases**: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión
- División: 50,000 imágenes de entrenamiento, 10,000 de test

### Tareas Implementadas

#### Cuestión 1 (2.5 pts)
- Creación de red convolucional usando la **API funcional** de Keras
- Arquitectura:
  - Capa de rescaling (normalización de píxeles a [0,1])
  - **3 capas convolucionales** (Conv2D) con 128 filtros cada una
  - **3 capas de pooling** (MaxPooling2D) de tamaño 2x2
  - Capa de aplanado (Flatten)
  - Capa densa con 128 neuronas
  - Capa de salida con 10 neuronas y activación softmax
- Función de pérdida: `SparseCategoricalCrossentropy` (apropiada para clasificación multiclase con etiquetas enteras)
- Métrica: `accuracy`
- Resultado: **Test Accuracy ≈ 0.703** (supera el requisito de >0.68)

#### Cuestión 2 (1 pt)
- Recreación del mismo modelo usando la **API secuencial**
- Misma arquitectura pero con estructura secuencial
- No requiere compilación ni entrenamiento

#### Cuestiones Teóricas (1.5 pts)
- **Cuestión 3**: Cálculo del número de parámetros en una capa densa para imágenes grandes
- **Cuestión 4**: Ventajas de las redes convolucionales frente a las densas
- **Cuestión 5**: Efectividad de las CNNs en series temporales

---

## Instrucciones de Ejecución

### Requisitos Previos
- Python 3.x
- TensorFlow/Keras
- Pandas, NumPy, Matplotlib
- Scikit-learn

### Opción Recomendada: Google Colab
Dado que el entrenamiento de redes neuronales es computacionalmente costoso, **se recomienda ejecutar el notebook en [Google Colab](https://colab.research.google.com)**:

1. Accede a [Google Colab](https://colab.research.google.com)
2. Haz clic en `Upload` y sube el archivo `Jacobo_Alvarez_Gutierrez.ipynb`
3. Ejecuta todas las celdas en orden
4. Descarga el notebook ejecutado: `File → Download → .ipynb`

**Importante**: El notebook debe entregarse con todas las celdas ejecutadas. Las celdas sin ejecutar no se contarán.

### Ejecución Local
Si prefieres ejecutarlo localmente:

```bash
# Instalar dependencias
pip install tensorflow pandas numpy matplotlib scikit-learn

# Ejecutar el notebook
jupyter notebook Jacobo_Alvarez_Gutierrez.ipynb
```

---

## Interpretación de Resultados

### Actividad 1: Redes Densas

#### Métricas de Evaluación
- **Loss (MSE)**: Error cuadrático medio. Valores más bajos indican mejor ajuste.
- **MAE**: Error absoluto medio. Representa la diferencia promedio entre predicciones y valores reales.

#### Comparación de Modelos
1. **Modelo sin regularización**: 
   - Test Loss: 0.559
   - Puede mostrar signos de overfitting (diferencia entre train y validation loss)

2. **Modelo con regularización**:
   - Test Loss: 0.536 (mejora del ~4%)
   - Las técnicas de regularización (L2 + Dropout) mejoran la generalización

3. **Modelo con Early Stopping**:
   - Test Loss: 0.523 (mejora adicional)
   - El early stopping previene el sobreentrenamiento deteniendo el entrenamiento cuando la pérdida de validación deja de mejorar

### Actividad 2: Redes Convolucionales

#### Métricas de Evaluación
- **Accuracy**: Porcentaje de imágenes clasificadas correctamente
- **Loss**: Entropía cruzada. Valores más bajos indican mejor clasificación.

#### Resultados del Modelo
- **Test Accuracy: 70.27%**: El modelo clasifica correctamente más del 70% de las imágenes de test
- La arquitectura convolucional permite extraer características espaciales de las imágenes
- El uso de pooling reduce la dimensionalidad y ayuda a la generalización

---

## Estructura del Notebook

El notebook está organizado de la siguiente manera:

1. **Configuración inicial**: Importación de librerías y configuración de semillas
2. **Actividad 1**: 
   - Carga y preprocesamiento del dataset de vinos
   - Normalización de features
   - Implementación de modelos densos progresivamente mejorados
   - Cuestiones teóricas sobre redes densas
3. **Actividad 2**:
   - Carga del dataset CIFAR-10
   - Visualización de imágenes
   - Implementación de red convolucional (API funcional y secuencial)
   - Cuestiones teóricas sobre CNNs

---

## Notas Técnicas

### Normalización
- **Actividad 1**: Se usa `Normalization` layer adaptada solo con datos de entrenamiento
- **Actividad 2**: Se usa `Rescaling` para normalizar píxeles de [0,255] a [0,1]

### Funciones de Activación
- **Capas ocultas**: ReLU (Rectified Linear Unit) - estándar para la mayoría de casos
- **Capa de salida (regresión)**: Sin activación (linear) - permite valores continuos
- **Capa de salida (clasificación)**: Softmax - convierte salidas en probabilidades

### Optimización
- Optimizador: Adam (Adaptive Moment Estimation)
- Batch size: Variable según el modelo (32 o 64)
- Épocas: 200 para redes densas, 25 para CNNs

---

## Conclusiones

Esta tarea demuestra:
- La importancia de la normalización en el preprocesamiento de datos
- El efecto de las técnicas de regularización en la generalización del modelo
- La eficacia de las redes convolucionales para tareas de visión por computadora
- La diferencia entre problemas de regresión y clasificación en deep learning

Los resultados muestran mejoras progresivas al aplicar técnicas de regularización y optimización, destacando la importancia de un diseño cuidadoso de la arquitectura y los hiperparámetros.

