# Ejercicios de Machine Learning con scikit-learn

## Estructura de Archivos

- **Archivos sin "jacobo_alvarez_gutierrez"**: Son los **enunciados** de las tareas (ej: `ejercicio1.ipynb`, `ejercicio2.ipynb`)
  - Contienen las instrucciones, código base y espacios para implementar la solución
  - Incluyen celdas de control con `assert` para validar las respuestas

- **Archivos con "jacobo_alvarez_gutierrez"**: Son las **soluciones completas** implementadas (ej: `ejercicio1_jacobo_alvarez_gutierrez.ipynb`)
  - Contienen todo el código resuelto y ejecutado
  - Todas las celdas de control pasan correctamente

---

## Resumen de Ejercicios

### Ejercicio 1: Modelo End-to-End (6 puntos)
- Dataset: Titanic
- Contenido: Transformación de datos, evaluación de algoritmos múltiples, tuneado con GridSearchCV
- Técnicas: ColumnTransformer, KNNImputer, OneHotEncoder, validación cruzada, RandomForest

### Ejercicio 2: Modelos y Transformaciones (14 puntos)
- Dataset: KDD Cup 99
- Contenido: Visualización, múltiples modelos con diferentes transformaciones, PCA, evaluación de overfitting
- Técnicas: Countplots, pipelines, PCA, GridSearchCV, análisis de matrices de confusión

---

## Ejecución

### Requisitos
- Python 3.10
- scikit-learn 1.5.2, matplotlib 3.9.2, seaborn 0.13.2, pandas 2.2.3, numpy 2.1.2

### Pasos
1. Abrir el notebook de solución en Jupyter
2. Ejecutar: `Cell -> Run All Cells` (o celda por celda con `Shift + Enter`)
3. Verificar que todas las celdas de control (`assert`) pasen sin errores

**Importante**: Ejecutar las celdas en orden, ya que cada una depende de las anteriores.

---

## Interpretación de Resultados

### Conceptos Clave
- **ColumnTransformer**: Aplica transformaciones diferentes a distintas columnas
- **Pipelines**: Encadenan transformadores y algoritmos
- **Validación Cruzada**: Evalúa modelos de forma robusta (KFold, StratifiedKFold)
- **GridSearchCV**: Optimiza hiperparámetros del modelo

### Métricas Importantes
- **Accuracy**: Precisión general del modelo
- **Confusion Matrix**: Muestra verdaderos/falsos positivos y negativos
- **Overfitting**: Diferencia entre `acc_train` y `acc_test` (si es grande, hay sobreajuste)
- **Boxplots**: Comparan el rendimiento de múltiples algoritmos

### Estructura de las Soluciones
1. **Inicialización**: Librerías, semillas, verificación de versiones
2. **Preparación de datos**: Carga, limpieza, transformaciones
3. **Modelado**: Evaluación y comparación de algoritmos
4. **Optimización**: Tuneado de hiperparámetros (GridSearchCV)

---

## Solución de Problemas

- **Errores en asserts**: Revisar la implementación en la sección correspondiente
- **Versiones incorrectas**: Instalar las versiones específicas requeridas
- **Variables no definidas**: Asegurarse de ejecutar todas las celdas en orden desde el principio

---

**Autor**: Jacobo Álvarez Gutiérrez
