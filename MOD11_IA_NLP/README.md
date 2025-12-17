# Tareas de NLP - Fine-tuning de Modelos Transformers

Este repositorio contiene las tareas de Procesamiento de Lenguaje Natural (NLP) realizadas por Jacobo Álvarez Gutiérrez, enfocadas en el fine-tuning de modelos transformers utilizando la biblioteca Hugging Face Transformers.

## 📁 Organización de Archivos

La estructura del repositorio sigue un patrón organizativo claro:

- **`nlp_tarea1.ipynb`** y **`nlp_tarea2.ipynb`**: Contienen los enunciados y plantillas de las tareas proporcionadas por el profesorado.
- **`nlp_tarea1_jacobo_alvarez_gutierrez.ipynb`** y **`nlp_tarea2_jacobo_alvarez_gutierrez.ipynb`**: Contienen las soluciones implementadas con el nombre del estudiante.

Esta organización permite distinguir claramente entre los enunciados originales y las soluciones desarrolladas.

## 📋 Resumen de Tareas

### Tarea 1: Classification Fine-tuning

**Enunciado:**
- **Objetivo**: Realizar fine-tuning de un modelo transformer para clasificación de texto utilizando el dataset **MNLI** (Multi-Genre Natural Language Inference) del benchmark **GLUE**.
- **Dataset**: El dataset MNLI consiste en pares de oraciones (premisa e hipótesis) con tres posibles relaciones:
  - **Entailment**: La hipótesis es una conclusión lógica de la premisa
  - **Neutral**: La hipótesis no puede determinarse como verdadera o falsa
  - **Contradiction**: La hipótesis contradice la premisa
- **Splits utilizados**:
  - `train`: 13,635 ejemplos (después del filtrado)
  - `validation_matched`: 413 ejemplos
  - `validation_mismatched`: 296 ejemplos
- **Filtrado**: Se utilizan solo registros con longitud de premisa inferior a 20 caracteres para reducir tiempos de entrenamiento.

**Solución implementada:**
- **Modelo base**: `textattack/bert-base-uncased-MNLI` (BERT pre-entrenado específicamente para MNLI)
- **Tokenizador**: `AutoTokenizer` del mismo modelo
- **Preprocesamiento**: Función `preprocess_function` que tokeniza premisa e hipótesis con truncamiento y padding, retornando tensores de PyTorch
- **Entrenamiento**: 
  - Learning rate: 2e-5
  - Batch size: 16 (train), 64 (eval)
  - Épocas: 4
  - Weight decay: 0.01
- **Resultados**: El modelo alcanzó una nota de 10/10, superando todos los umbrales requeridos:
  - `validation_matched`: Accuracy 0.85, Precision y Recall superiores a los umbrales
  - `validation_mismatched`: Accuracy 0.84, métricas consistentes

### Tarea 2: Question Answering Fine-tuning

**Enunciado:**
- **Objetivo**: Realizar fine-tuning de un modelo transformer para tareas de Question Answering (QA) utilizando el dataset **SQuAD** (Stanford Question Answering Dataset).
- **Dataset**: SQuAD consiste en ternas de preguntas, respuestas y contexto. El modelo debe extraer la respuesta del contexto dado.
- **Splits utilizados**:
  - `train`: 3,466 ejemplos (después del filtrado)
  - `validation`: 345 ejemplos
- **Filtrado**: Se utilizan solo registros con longitud de contexto inferior a 300 caracteres.

**Solución implementada:**
- **Modelo base**: `deepset/roberta-base-squad2` (RoBERTa pre-entrenado para SQuAD 2.0)
- **Tokenizador**: `AutoTokenizer` del mismo modelo
- **Preprocesamiento**: Función `preprocess_function` compleja que:
  - Tokeniza pregunta y contexto con truncamiento solo del contexto
  - Maneja overflow de tokens con stride de 256
  - Calcula posiciones de inicio y fin de las respuestas en el espacio de tokens
  - Maneja casos sin respuesta asignando posiciones al token CLS
- **Entrenamiento**:
  - Learning rate: 3e-5
  - Batch size: 8 (train), 16 (eval)
  - Gradient accumulation: 2 pasos
  - Épocas: 4
  - Weight decay: 0.01
  - Warmup ratio: 0.1
  - Max gradient norm: 1.0
- **Evaluación**: Se evalúa sobre 20 muestras específicas del conjunto de validación usando similitud de palabras (Jaccard-like)
- **Resultados**: El modelo alcanzó una nota de 10/10, demostrando alta precisión en la extracción de respuestas

## 🚀 Instrucciones de Ejecución

### Requisitos Previos

1. **Entorno Python**: Se recomienda usar un entorno virtual (conda o venv)
2. **CUDA**: Los notebooks están configurados para usar GPU si está disponible (CUDA 12.8 en el entorno de desarrollo)
3. **Dependencias principales**:
   - `torch` (PyTorch)
   - `transformers` (Hugging Face)
   - `datasets` (Hugging Face)
   - `scikit-learn`
   - `matplotlib`
   - `pandas`
   - `numpy`

### Pasos para Ejecutar

1. **Instalar dependencias**:
```bash
pip install torch transformers datasets scikit-learn matplotlib pandas numpy
```

2. **Ejecutar los notebooks**:
   - Abrir Jupyter Notebook o JupyterLab
   - Ejecutar las celdas en orden secuencial
   - **Importante**: No modificar las celdas marcadas con `# No modificar esta celda`

3. **Tiempo estimado de ejecución**:
   - **Tarea 1**: Aproximadamente varios minutos (depende del hardware)
   - **Tarea 2**: Similar a la Tarea 1

### Notas Importantes

- Los notebooks de solución (`*_jacobo_alvarez_gutierrez.ipynb`) contienen código ejecutado y resultados guardados
- Las celdas marcadas como "No modificar" son parte del sistema de evaluación y deben ejecutarse tal cual
- Los modelos se descargarán automáticamente desde Hugging Face Hub en la primera ejecución
- Los checkpoints de entrenamiento se guardan en las carpetas `./results1/` y `./results2/` respectivamente

### Verificación de Ejecución Correcta

**Tarea 1:**
- El notebook debe ejecutarse sin errores
- Debe mostrar métricas de accuracy, precision y recall para los tres splits
- Debe mostrar matrices de confusión
- Debe calcular una nota final

**Tarea 2:**
- El notebook debe ejecutarse sin errores
- Debe crear un pipeline de question-answering
- Debe evaluar sobre las 20 muestras especificadas
- Debe mostrar un DataFrame con predicciones y métricas de match
- Debe calcular una nota final

## 📊 Resultados

Ambas tareas fueron completadas exitosamente con notas máximas (10/10), demostrando:
- Comprensión adecuada de los modelos transformers
- Implementación correcta de pipelines de fine-tuning
- Manejo apropiado de preprocesamiento de datos
- Configuración efectiva de hiperparámetros

---

**Autor**: Jacobo Álvarez Gutiérrez  
**Módulo**: MOD11 - Inteligencia Artificial - NLP

