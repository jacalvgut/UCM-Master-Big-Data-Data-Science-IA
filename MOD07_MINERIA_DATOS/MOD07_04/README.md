# Tarea de Evaluación: Minería de Datos y Modelización Predictiva

## Resumen del Enunciado

Esta tarea consiste en un **cuestionario tipo test** compuesto por 15 preguntas, valorado sobre 10 puntos. El cuestionario evalúa conocimientos sobre minería de datos y modelización predictiva, con especial énfasis en modelos de scoring crediticio y análisis espacial.

### Características Generales

- **Formato**: Cuestionario tipo test con 15 preguntas
- **Valoración**: Cada pregunta vale 1 punto, pero la nota final se calcula sobre 10 puntos mediante reponderación (10/15 puntos por pregunta)
- **Intentos**: Se disponen de **2 intentos** para completar el cuestionario
- **Tiempo**: No hay restricción de tiempo, solo fecha límite de entrega
- **Penalización**: Las respuestas incorrectas pueden penalizar hasta -1 punto, repartido entre el número de posibles respuestas incorrectas

### Estructura del Cuestionario

El cuestionario incluye diferentes tipos de preguntas:

1. **Preguntas teóricas** (1-7, 10-14): Cubren conceptos generales de gestión de riesgos, minería de datos y modelización predictiva.

2. **Preguntas prácticas con datos de Scoring** (8-9): Requieren trabajar con el archivo `DatosPractica_Scoring.xlsx`:
   - **Pregunta 8**: Cálculo del estadístico Valor de Información (IV) para una muestra específica de datos
   - **Pregunta 9**: Construcción completa de un modelo de puntuación crediticia, incluyendo:
     - Análisis exploratorio y depuración de datos
     - Inferencia de rechazados (considerar tanto clientes aceptados como rechazados)
     - División de muestra para entrenamiento y test
     - Pronóstico para nuevos clientes (IDs 1286-1319)

3. **Pregunta práctica de Análisis Espacial** (15): Requiere trabajar con el archivo `Data_Housing_Madrid.csv`:
   - Análisis de autocorrelación espacial en precios de viviendas del centro histórico de Madrid
   - Cálculo de matrices de pesos espaciales
   - Estimación del índice I de Moran

### Datos para las Preguntas Prácticas

#### Datos de Scoring (`DatosPractica_Scoring.xlsx`)

**Variable objetivo:**
- `Default`: Indica si un cliente ha hecho impago (1 = impago, 0 = no impago)

**Variables de clasificación:**
- `Cardhldr = 1`: Clientes aceptados (con tarjeta concedida)
- `Cardhldr = 0`: Clientes rechazados (solicitud denegada)
- `Cardhldr = na`: Nuevos solicitantes (IDs 1286-1319) a puntuar

**Variables explicativas disponibles:**
- `ID`: Identificador del solicitante
- `Age`: Edad en años más fracciones de año
- `Income`: Ingresos anuales (divididos por 10,000)
- `Exp_Inc`: Ratio de gasto mensual en tarjeta de crédito sobre ingresos anuales
- `Avgexp`: Gasto mensual promedio en tarjeta de crédito
- `Ownrent`: 1 si es propietario, 0 si alquila
- `Selfempl`: 1 si es autónomo, 0 si no
- `Depndt`: Número de personas a cargo
- `Inc_per`: Ingresos divididos por (1 + número de dependientes)
- `Cur_add`: Meses viviendo en la dirección actual
- `Major`: Número de tarjetas de crédito principales
- `Active`: Número de cuentas de crédito activas

#### Datos de Viviendas (`Data_Housing_Madrid.csv`)

**Variables de interés:**
- `house.price`: Precio de la vivienda (euro/m²)
- `historical`: Indica si pertenece al casco histórico de Madrid
- `longitude`: Longitud en coordenadas geográficas
- `latitude`: Latitud en coordenadas geográficas

**Objetivo**: Cuantificar la autocorrelación espacial en el precio por metro cuadrado del centro histórico de Madrid.

### Consideraciones Importantes

- **Decimales**: En Windows con teclado español, utilizar **coma (,) como separador decimal**, no punto (.)
- **Respuestas múltiples**: Para obtener el punto completo, es necesario marcar **todas** las respuestas correctas
- **Validación**: El sistema indicará si hay error al pasar a la siguiente pregunta si el formato numérico es incorrecto

---

## Guía de Lectura del Documento de Solución

El documento `jacobo_alvarez_gutierrez.pdf` contiene la revisión del intento realizado del cuestionario, con una calificación final de **8,31 sobre 10,00 puntos (83%)**.

### Estructura del Documento

El documento muestra:

1. **Información general del intento**:
   - Fecha de inicio y finalización
   - Tiempo empleado
   - Puntuación obtenida
   - Calificación final

2. **Revisión pregunta por pregunta**:
   - Cada pregunta muestra:
     - Estado de corrección (Correcta, Parcialmente correcta, Incorrecta)
     - Puntuación obtenida sobre el máximo posible
     - Las respuestas seleccionadas (marcadas con ☑ o ☐)
     - En algunos casos, las respuestas correctas según el sistema

### Cómo Interpretar los Resultados

#### Preguntas Parcialmente Correctas

Cuando una pregunta aparece como "Parcialmente correcta" (por ejemplo, 0,80 sobre 1,00), significa que:
- Se marcaron algunas respuestas correctas, pero no todas
- En preguntas de respuesta múltiple, cada respuesta correcta tiene un valor igual a 1 dividido por el número total de respuestas correctas
- Es necesario revisar qué respuestas faltaron por marcar

**Ejemplo del documento**: La Pregunta 1 muestra 0,80/1,00, indicando que faltó marcar al menos una respuesta correcta de las múltiples opciones disponibles.

#### Preguntas Correctas

Las preguntas marcadas como "Se puntúa 1,00 sobre 1,00" indican que:
- Todas las respuestas correctas fueron seleccionadas
- No se marcaron respuestas incorrectas (o si se marcaron, la penalización no afectó el resultado final)

#### Valores Numéricos en Preguntas Prácticas

Para las preguntas que requieren valores numéricos (especialmente la pregunta 15):
- Los valores mostrados con ☑ indican respuestas correctas
- Los valores con ☐ indican respuestas que no fueron seleccionadas o que eran incorrectas
- Prestar atención a los decimales y redondeos solicitados

**Ejemplo del documento - Pregunta 15**:
- Número total de viviendas: 10512 ☑
- Viviendas en casco histórico: 3633 ☑
- Precio mediano: 4444 ☑
- Precio máximo: 20131 ☑
- Viviendas sin vecinas: 106 ☑
- Número mediano de vecinas: 14 ☐ (esta respuesta no fue seleccionada o era incorrecta)
- I de Moran: 0,326 ☑
- p-valor: 0,001 ☑

### Recomendaciones para la Lectura

1. **Identificar patrones de error**: Revisar qué tipos de preguntas tuvieron menor puntuación para identificar áreas de mejora

2. **Validar respuestas numéricas**: Para las preguntas prácticas (8, 9, 15), comparar los valores obtenidos con los mostrados en el documento para verificar la metodología

3. **Revisar respuestas múltiples**: En preguntas teóricas con múltiples opciones correctas, verificar que se marcaron todas las respuestas necesarias

4. **Aprender de las correcciones**: El documento puede incluir comentarios o indicaciones sobre las respuestas correctas, útiles para el segundo intento

5. **Nota sobre el segundo intento**: Si se realiza un segundo intento, es recomendable anotar las preguntas que estaban correctas en el primer intento, ya que las respuestas pueden no mantenerse marcadas automáticamente

### Aspectos Técnicos Destacados

El documento muestra que se trabajó correctamente con:
- **Análisis espacial**: Cálculo de matrices de pesos espaciales usando `DistanceBand` con umbral de 0,00225 grados (aproximadamente 250 metros)
- **Estadísticos espaciales**: Cálculo del índice I de Moran y su p-valor
- **Filtrado de datos**: Selección correcta de viviendas del casco histórico

---

## Notas Finales

- La nota final del cuestionario será la **mayor de las dos notas** obtenidas en cada intento
- Se recomienda guardar un intento hasta después de la tutoría online para poder resolver dudas
- Una vez cerrado el cuestionario, se puede acceder para revisar las respuestas correctas

---

*Documento elaborado para facilitar la comprensión del enunciado de la tarea y la interpretación de los resultados obtenidos.*

