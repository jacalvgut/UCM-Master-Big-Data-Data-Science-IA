# Tarea: Redes Neuronales Recurrentes (RNN)

## Resumen del Enunciado

Esta tarea consiste en la implementación y evaluación de redes neuronales recurrentes para la predicción de series temporales. El objetivo principal es predecir la temperatura mínima diaria en Melbourne con un horizonte de predicción de 2 días, utilizando diferentes arquitecturas y técnicas de ingeniería de características.

### Objetivos de Aprendizaje

- Implementar ventanas deslizantes para series temporales
- Construir modelos recurrentes con capas GRU
- Mejorar modelos mediante la incorporación de características adicionales
- Comprender arquitecturas many-to-one y sus aplicaciones
- Entender las ventajas de los word embeddings en procesamiento de lenguaje natural

## Estructura de la Solución

La solución está organizada en **5 cuestiones** que abordan diferentes aspectos del problema:

### Actividad 1: Redes Recurrentes (10 puntos)

#### Cuestión 1 (2.5 puntos): Creación de Ventanas Temporales
- **Objetivo**: Convertir los datos de entrenamiento y prueba en ventanas de tamaño 5 para predecir valores con un horizonte de 2 días.
- **Parámetros**: `past=5`, `future=2`
- **Implementación**: Función `create_windows()` que genera secuencias de entrada (X) y valores objetivo (y)
- **Interpretación**: Cada ventana contiene 5 valores consecutivos de temperatura, y el objetivo es predecir la temperatura que ocurrirá 2 días después del último valor de la ventana

#### Cuestión 2 (2.5 puntos): Modelo GRU Básico
- **Arquitectura**: Modelo de dos capas GRU con:
  - Capa de normalización de entrada
  - Primera capa GRU: 64 unidades, `return_sequences=True` (para pasar secuencias a la siguiente capa)
  - Segunda capa GRU: 64 unidades (retorna solo el estado final)
  - Capa densa de salida con 1 neurona
  - Dropout y recurrent dropout del 20% para regularización
- **Métricas**: MSE (Mean Squared Error) y MAE (Mean Absolute Error)
- **Resultado**: El modelo alcanza un MAE de aproximadamente 2.05 en el conjunto de prueba

#### Cuestión 3 (2.5 puntos): Mejora con Features Adicionales
- **Características añadidas**:
  - `portion_year`: Porción del año (día del año / 365)
  - Codificación cíclica mediante seno y coseno para:
    - Año (`year_sin`, `year_cos`)
    - Mes (`month_sin`, `month_cos`)
    - Día de la semana (`dow_sin`, `dow_cos`)
- **Mejoras del modelo**:
  - Aumento de unidades GRU a 128 (mayor capacidad)
  - Entrada expandida de 1 a 8 características
- **Resultado**: Mejora significativa con MAE de aproximadamente 1.95 en el conjunto de prueba

#### Cuestión 4 (1.25 puntos): Arquitecturas Many-to-One
- **Pregunta teórica** sobre aplicaciones de arquitecturas many-to-one
- **Respuesta**: Aplicable en clasificación de sentimiento, verificación de voz y clasificación de música por autor
- **No aplicable**: Generación de música (requiere many-to-many)

#### Cuestión 5 (1.25 puntos): Word Embeddings
- **Pregunta teórica** sobre ventajas de word embeddings
- **Respuesta**: Todas las opciones son correctas:
  - Reducción de dimensionalidad vs. one-hot encoding
  - Captura de similitud semántica entre palabras
  - Transfer learning en NLP
  - Visualización mediante técnicas de reducción de dimensionalidad

## Guía de Lectura e Interpretación

### Orden Recomendado de Lectura

1. **Celdas iniciales (0-6)**: Configuración del entorno y carga de datos
   - Importación de librerías necesarias
   - Descarga y carga del dataset de temperaturas
   - División en conjuntos de entrenamiento (3000 muestras) y prueba (650 muestras)

2. **Cuestión 1 (Celdas 7-9)**: Entender la función `create_windows()`
   - Analizar cómo se generan las ventanas deslizantes
   - Verificar el ejemplo con las primeras 10 muestras
   - Comprender la relación entre `past`, `future` y los índices de las ventanas

3. **Cuestión 2 (Celdas 10-13)**: Arquitectura del modelo básico
   - Revisar la estructura de capas GRU
   - Entender el uso de `return_sequences=True` en la primera capa
   - Analizar los resultados del entrenamiento y evaluación
   - Observar la mejora progresiva de las métricas durante el entrenamiento

4. **Cuestión 3 (Celdas 14-19)**: Mejora del modelo
   - Estudiar las características temporales añadidas
   - Comprender la codificación cíclica (seno/coseno) para variables temporales
   - Comparar los resultados con el modelo anterior
   - Notar la mejora en MAE de ~2.05 a ~1.95

5. **Cuestiones 4 y 5 (Celdas 20-23)**: Conceptos teóricos
   - Revisar las respuestas y justificaciones
   - Relacionar con conceptos vistos en clase

### Puntos Clave para la Interpretación

#### Ventanas Temporales
- **Concepto**: Las ventanas deslizantes permiten convertir una serie temporal unidimensional en un problema de aprendizaje supervisado
- **Parámetros importantes**:
  - `window_size` (past): Número de pasos temporales usados como entrada
  - `horizon` (future): Cuántos pasos hacia adelante se predice
- **Ejemplo visual**: Para `past=5, future=2`:
  ```
  Input:  [t₀, t₁, t₂, t₃, t₄]
  Output: t₆  (2 días después de t₄)
  ```

#### Arquitectura GRU
- **Ventajas sobre LSTM**: Menor complejidad computacional manteniendo capacidad de capturar dependencias temporales
- **Dropout**: Previene sobreajuste durante el entrenamiento
- **Normalización**: Facilita la convergencia del modelo

#### Features Temporales
- **Codificación cíclica**: Las variables temporales (mes, día de semana) son cíclicas. Usar seno/coseno permite que el modelo entienda que diciembre está cerca de enero
- **Beneficio**: El modelo puede aprender patrones estacionales más efectivamente

#### Comparación de Modelos
- **Modelo básico**: Solo temperatura → MAE ~2.05
- **Modelo mejorado**: Temperatura + features temporales → MAE ~1.95
- **Mejora relativa**: ~5% de reducción en error absoluto medio

### Notas Técnicas

- **Entrenamiento**: Se recomienda ejecutar en Google Colab debido al costo computacional
- **Early Stopping**: Implementado con paciencia de 10 épocas para evitar sobreentrenamiento
- **División de datos**: 3000 muestras para entrenamiento, 650 para prueba
- **Normalización**: Adaptada solo con datos de entrenamiento para evitar data leakage

## Dependencias

```python
tensorflow >= 2.x
keras
matplotlib
pandas
numpy
```

## Ejecución

1. Subir el notebook a Google Colab o ejecutar localmente
2. Ejecutar las celdas en orden secuencial
3. Asegurarse de que todas las celdas estén ejecutadas para la entrega

## Resultados Esperados

- **Cuestión 1**: Generación correcta de ventanas con las dimensiones apropiadas
- **Cuestión 2**: Modelo GRU entrenado con MAE de validación alrededor de 2.0-2.3
- **Cuestión 3**: Modelo mejorado con MAE de validación alrededor de 1.9-2.0
- **Cuestiones 4 y 5**: Respuestas teóricas correctamente justificadas

