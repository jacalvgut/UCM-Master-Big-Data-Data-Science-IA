# Tarea Final de Estadística
## Análisis Comparativo de Anchuras de Cráneo entre Períodos Históricos

**Autor:** Jacobo Álvarez Gutiérrez

---

## Enunciado de la Tarea

Esta tarea consiste en realizar un análisis estadístico comparativo de las anchuras de cráneo de dos períodos históricos diferentes:

- **Predinástico Temprano** (Época histórica 1)
- **Predinástico Tardío** (Época histórica 2)

### Objetivos del Análisis

1. **Ejercicio 1**: Análisis descriptivo y verificación de normalidad
   - Calcular medidas estadísticas descriptivas para ambos períodos
   - Realizar tests de normalidad (Kolmogorov-Smirnov y Shapiro-Wilk)
   - Generar visualizaciones (histogramas y diagramas de caja y bigotes)

2. **Ejercicio 2**: Comparación de muestras e inferencia estadística
   - Comparar varianzas poblacionales entre ambos períodos
   - Calcular intervalos de confianza para la diferencia de medias
   - Realizar tests de hipótesis para comparar las medias de ambas muestras
   - Aplicar métodos alternativos (Bootstrap, Mann-Whitney U)

---

## Estructura de Archivos

### 📓 `alvarez_gutierrez_jacobo_tarea_final.ipynb`

Este notebook de Jupyter contiene toda la solución implementada en Python. Está organizado en las siguientes secciones:

#### **Carga de Datos y Preparación**
- Importación de librerías necesarias (pandas, numpy, matplotlib, seaborn, scipy, sklearn)
- Lectura del archivo Excel con los datos (`datosejercicioevaluacionanchuras.xlsx`)
- Separación de datos por período histórico (Predinástico Temprano y Predinástico Tardío)

#### **Ejercicio 1 - Análisis Descriptivo**
- **Función `analizar_periodos()`**: Calcula y presenta un resumen estadístico completo que incluye:
  - Medidas de tendencia central (media, mediana, moda)
  - Medidas de dispersión (desviación estándar, varianza, rango)
  - Cuartiles (Q1, Q2, Q3)
  - Coeficientes de asimetría (Pearson, Fisher) y curtosis
  - Generación de histogramas y diagramas de caja y bigotes

#### **Ejercicio 1 - Tests de Normalidad**
- **Test de Kolmogorov-Smirnov**: Verifica si las muestras siguen una distribución normal
- **Test de Shapiro-Wilk**: Método alternativo para verificar normalidad (más robusto para muestras pequeñas)

#### **Ejercicio 2 - Comparación de Varianzas**
- **Test F**: Compara las varianzas poblacionales de ambas muestras
- **Test de Levene**: Método alternativo para verificar homogeneidad de varianzas
- Visualización gráfica de la distribución F con valores críticos

#### **Ejercicio 2 - Intervalos de Confianza**
- Cálculo de intervalos de confianza para la diferencia de medias al 90%, 95% y 99%
- **Método Bootstrap**: Implementación alternativa para calcular intervalos de confianza mediante remuestreo

#### **Ejercicio 2 - Tests de Hipótesis**
- **Test t de Student**: Compara las medias de ambas muestras asumiendo normalidad y homogeneidad de varianzas
- **Test de Mann-Whitney U**: Método no paramétrico alternativo para comparar medianas

### 📄 `tarea_final_jacobo_alvarez_gutierrez.pdf`

Documento PDF que contiene:
- El enunciado completo de la tarea
- La solución desarrollada con explicaciones teóricas
- Resultados y conclusiones del análisis estadístico
- Interpretación de los resultados obtenidos

---

## Guía de Lectura

### Para entender la solución completa:

1. **Inicie con el PDF** (`tarea_final_jacobo_alvarez_gutierrez.pdf`)
   - Lea el enunciado completo para comprender el contexto y los objetivos
   - Revise las explicaciones teóricas de cada método estadístico utilizado
   - Consulte las conclusiones e interpretaciones de los resultados

2. **Explore el Notebook** (`alvarez_gutierrez_jacobo_tarea_final.ipynb`)
   - Ejecute las celdas en orden para reproducir el análisis
   - Observe los resultados numéricos y gráficos generados
   - Revise el código para entender la implementación de cada método

### Orden recomendado de lectura:

1. **Celdas 0-1**: Carga de datos y preparación del entorno
2. **Celda 2**: Análisis descriptivo completo con visualizaciones
3. **Celdas 3-4**: Tests de normalidad (Kolmogorov-Smirnov y Shapiro-Wilk)
4. **Celdas 5-8**: Comparación de varianzas e intervalos de confianza
5. **Celdas 9-10**: Tests de hipótesis (t de Student y Mann-Whitney U)

---

## Requisitos para Ejecutar el Notebook

Para ejecutar el notebook, asegúrese de tener instaladas las siguientes librerías de Python:

```python
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
openpyxl  # Para leer archivos Excel
```

Puede instalarlas usando:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
```

**Nota**: El notebook requiere el archivo de datos `datosejercicioevaluacionanchuras.xlsx` en el mismo directorio para funcionar correctamente.

---

## Resultados Principales

- **Normalidad**: Los tests indican que las muestras no siguen completamente una distribución normal
- **Varianzas**: No se encontraron diferencias significativas entre las varianzas de ambos períodos
- **Diferencia de medias**: Los intervalos de confianza muestran diferencias significativas entre las anchuras de cráneo de ambos períodos históricos
- **Tests de hipótesis**: Se confirma estadísticamente que existen diferencias significativas entre las medias de ambos períodos

---

## Contacto

Para consultas sobre esta tarea, contactar a: Jacobo Álvarez Gutiérrez

