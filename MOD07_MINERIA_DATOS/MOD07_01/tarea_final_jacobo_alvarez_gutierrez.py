# -*- coding: utf-8 -*-

# PROYECTO FINAL DE MINERÍA DE DATOS Y MODELIZACIÓN PREDICTIVA
# Jacobo Álvarez Gutiérrez

#####################
# DEPURACIÓN DE DATOS
#####################

# Importamos las librerías necesarias para este apartado
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import itertools
from collections import Counter

# Establecemos el directorio de trabajo
os.chdir('C:/Users/jacob/Desktop/master_ucm/mod7_1_mineria_modpredic/Tarea')

#Funciones propias del módulo de minería
from FuncionesMineria import (cuentaDistintos, analizar_variables_categoricas,
                              atipicosAmissing, ImputacionCuali, ImputacionCuant,
                              graficoVcramer, Rsq, crear_data_modelo,
                              lm_forward, lm_backward, lm_stepwise,
                              validacion_cruzada_lm, comparar_modelos,
                              modelEffectSizes, pseudoR2, summary_glm,
                              validacion_cruzada_glm, sensEspCorte, curva_roc,
                              glm_forward, glm_backward, glm_stepwise)

# Importamos el conjunto de datos
datos = pd.read_excel('DatosEleccionesEspaña.xlsx')

# En primer lugar vamos a eliminar las variables objetivo que no se han 
# seleccionado para este estudio.
var_obj_descartadas = ['Izda_Pct', 'Dcha_Pct', 'Otros_Pct',
                       'Izquierda', 'Derecha']
for i in var_obj_descartadas:
    datos.drop(i, axis=1, inplace=True)
    
# A continuación distinguimos las variables numéricas de las categóricas.
datos.dtypes

# Ejecutando el comando anterior notamos que en la asignación por defecto de
# Python, "AbstencionAlta" y "CodigoProvincia" se han asignado como variables
# numéricas erróneamente. La redefinimos como categórica y separamos ambos tipos
# de variables.
datos['AbstencionAlta'] = datos['AbstencionAlta'].astype(str)
datos['CodigoProvincia'] = datos['CodigoProvincia'].astype(str)

variables = list(datos.columns) 

numericas = datos.select_dtypes(include=['int', 'int32', 'int64', 'float',
                                         'float32', 'float64']).columns

categoricas = [variable for variable in variables if variable not in numericas]

# Comprobamos la efectividad de las modificaciones
datos.dtypes

# Con este comando comprobamos que ninguna de las variables numéricas tiene tan
# pocos valores como para considerarla categórica
cuentaDistintos(datos)

# Con esto comprobamos los diferentes valores de las variables categóricas. Nos
# sirve, entre otras cosas, para ver errores como datos missing.
analizar_variables_categoricas(datos)

# Analizando los resultados de la función anterior, vamos a recategorizar la variable
# 'ActividadPpal', ya que los campos "Construcción" e "Industria" están poco representados
# También debemos corregir valores missing (?) en la variable 'Densidad'

# Buscamos ahora posibles errores en las variables numéricas a partir de los
# estadísticos principales de la función describe() de Python y algún otro que
# calculamos por separado
descriptivos_num = datos.describe().T

for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)

# Corregimos los errores encontrados en las variables categóricas y numéricas

datos['Densidad'] = datos['Densidad'].replace('?', np.nan)
datos['ActividadPpal'] = datos['ActividadPpal'].replace({'Construccion': 'Otro',
                                                         'Industria': 'Otro'})

# En las variables numéricas, se ha observado que muchas de ellas representan
# porcentajes pero tienen valores fuera del rango [0, 100]. Por otra parte, el
# campo "explotaciones" presenta valores missing (99999)

ptge_outofrange = ['Age_19_65_pct', 'Age_over65_pct', 'ForeignersPtge',
                   'SameComAutonPtge']
for i in ptge_outofrange:
    datos[i] = [x if 0 <= x <= 100 else np.nan for x in datos[i]]
    
datos['Explotaciones'] = datos['Explotaciones'].replace(99999, np.nan)

#Verifico la efectividad de los cambios realizados
descriptivos_num = datos.describe().T

for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)

# Indico la variableObj, el ID y las Input. Guardo, además, los nombres de las
# distintas variables diferenciando también las categóricas de las numéricas.

datos = datos.set_index(datos['Name']).drop('Name', axis = 1)
varObjCont = datos['AbstentionPtge']
varObjBin = datos['AbstencionAlta']
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1)

variables_input = list(datos_input.columns)  

numericas_input = datos_input.select_dtypes(include = ['int', 'int32','int64',
                                                       'float', 'float32',
                                                       'float64']).columns

categoricas_input = [variable for variable in variables_input
                     if variable not in numericas_input]

# Datos atípicos

resultados = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input)
              for x in numericas_input}

# Vemos que la proporción de datos atípicos en cada una de las variables es pequeña.
# Por tanto, podemos considerarlos valores atípicos y transformarlos a valores missing.

for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]

# Valores missing

# Proporción de valores perdidos por cada variable (guardo la información)

prop_missingsVars = datos_input.isna().sum()/len(datos_input)

# Vemos que ninguna de las variables tiene la mitad de los datos missing así que no
# descartaremos ninguna de las variables en nuestro estudio

# Proporción de variables perdidas por cada observación

datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)
datos_input['prop_missings'].describe()

# De nuevo, vemos que ninguna de las observaciones tiene menos de la mitad de las
# variables, así que tampoco descartaremos ninguna.

# Veamos ahora si esta nueva variable 'prop_missings' debería ser una variable numérica
# o categórica

len(datos_input['prop_missings'].unique())
datos_input['prop_missings'].unique()

# Como solo hay 9 valores diferentes, consideraremos esta variable como categórica.
# No obstante, algunas de estas categorías están poco representadas, así que vamos
# a recategorizarla:
    
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)
datos_input["prop_missings"] = datos_input['prop_missings'].replace({'0.0': '0',
                                                                     '0.030303030303030304': '1',
                                                                     '0.06060606060606061': '1',
                                                                     '0.09090909090909091': '1',
                                                                     '0.12121212121212122': '1',
                                                                     '0.15151515151515152': '1',
                                                                     '0.18181818181818182': '1',
                                                                     '0.30303030303030304': '1',
                                                                     '0.3333333333333333': '1'})

# Verificamos de nuevo la efectividad de la recategorización llevada a cabo
len(datos_input['prop_missings'].unique())
datos_input['prop_missings'].unique()
analizar_variables_categoricas(datos_input)

# Agrego 'prop_missings' a la lista de nombres de variables input
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')

# Puesto que ninguna variable categórica tiene suficientes valores missing, no será necesario
# recategorizar. En su lugar, el único tratamiento que llevaremos a cabo para los
# datos missing será el de imputación.

for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'aleatorio')
    
for x in numericas_input:
    datos_input[x] = ImputacionCuant(datos_input[x], 'aleatorio')

# Comprobamos que no queden datos missing

datos_input.isna().sum()

# Terminamos aquí el proceso de depuración de los datos. Guardamos los datos
# depurados en otro archivo diferente.

datosEleccionesDep = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datosEleccionesDep.pickle', 'wb') as archivo:
    pickle.dump(datosEleccionesDep, archivo)

##########################
# RELACIÓN ENTRE VARIABLES
##########################

# Seguimos en el mismo directorio solo que ahora partiremos de los datos ya depurados

with open('datosEleccionesDep.pickle', 'rb') as f:
    datos = pickle.load(f)

varObjCont = datos['AbstentionPtge']
varObjBin = datos['AbstencionAlta']
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1) 

# Coeficiente de correlación de Pearson

numericas = datos_input.select_dtypes(include=['int', 'float']).columns
matriz_corr = pd.concat([varObjCont, datos_input[numericas]],
                        axis = 1).corr(method = 'pearson')
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
plt.figure(figsize=(22, 22))
sns.set(font_scale=1.2)
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f",
            cbar=True, mask=mask)
plt.title("Matriz de correlación")
plt.show()

# Estadístico V de Cramer

graficoVcramer(datos_input, varObjCont)
graficoVcramer(datos_input, varObjBin)

# De los resultados obtenidos concluimos que una primera selección de variables
# de cara a empezar a modelar las variables objetivo es la siguiente:

var_cont = ['totalEmpresas', 'TotalCensus', 'PersonasInmueble',
             'Age_0-4_Ptge', 'Age_under19_Ptge', 'Age_over65_pct']
var_categ = ['CCAA', 'CodigoProvincia', 'ActividadPpal']

#############################
# MODELOS DE REGRESIÓN LINEAL
#############################

# Hacemos una partición train/test del conjunto de datos

x_train, x_test, y_train, y_test = train_test_split(datos_input,
                                                    np.ravel(varObjCont),
                                                    test_size = 0.2,
                                                    random_state = 123456)

# Métodos de selección clásica de variables

# ModeloStepwise(AIC)

modeloStepAIC = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloStepAIC['Modelo'].summary()
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], 
                                                modeloStepAIC['Variables']['categ'], 
                                                modeloStepAIC['Variables']['inter'])
Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)

# ModeloStepwise(BIC)

modeloStepBIC = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'BIC')
modeloStepBIC['Modelo'].summary()
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], 
                                                modeloStepBIC['Variables']['categ'], 
                                                modeloStepBIC['Variables']['inter'])
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)

# ModeloBackward(AIC)

modeloBackAIC = lm_backward(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloBackAIC['Modelo'].summary()
Rsq(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X'])
x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], 
                                                modeloBackAIC['Variables']['categ'], 
                                                modeloBackAIC['Variables']['inter'])
Rsq(modeloBackAIC['Modelo'], y_test, x_test_modeloBackAIC)

# ModeloBackward(BIC)

modeloBackBIC = lm_backward(y_train, x_train, var_cont, var_categ, [], 'BIC')
modeloBackBIC['Modelo'].summary()
Rsq(modeloBackBIC['Modelo'], y_train, modeloBackBIC['X'])
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], 
                                                modeloBackBIC['Variables']['categ'], 
                                                modeloBackBIC['Variables']['inter'])
Rsq(modeloBackBIC['Modelo'], y_test, x_test_modeloBackBIC)

# ModeloForward(AIC)

modeloForAIC = lm_forward(y_train, x_train, var_cont, var_categ, [], 'AIC')
modeloForAIC['Modelo'].summary()
Rsq(modeloForAIC['Modelo'], y_train, modeloForAIC['X'])
x_test_modeloForAIC = crear_data_modelo(x_test, modeloForAIC['Variables']['cont'], 
                                                modeloForAIC['Variables']['categ'], 
                                                modeloForAIC['Variables']['inter'])
Rsq(modeloForAIC['Modelo'], y_test, x_test_modeloForAIC)

# ModeloForward(BIC)

modeloForBIC = lm_forward(y_train, x_train, var_cont, var_categ, [], 'BIC')
modeloForBIC['Modelo'].summary()
Rsq(modeloForBIC['Modelo'], y_train, modeloForBIC['X'])
x_test_modeloForBIC = crear_data_modelo(x_test, modeloForBIC['Variables']['cont'], 
                                                modeloForBIC['Variables']['categ'], 
                                                modeloForBIC['Variables']['inter'])
Rsq(modeloForBIC['Modelo'], y_test, x_test_modeloForBIC)

# Vamos a probar ahora con interacciones

interacciones = list(itertools.combinations(var_cont, 2))

# ModeloStepwise(AIC) con interacciones

modeloStepAIC_int = lm_stepwise(y_train, x_train, var_cont, var_categ,
                                interacciones, 'AIC')
modeloStepAIC_int['Modelo'].summary()
Rsq(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])
x_test_modeloStepAIC_int = crear_data_modelo(x_test, modeloStepAIC_int['Variables']['cont'], 
                                                modeloStepAIC_int['Variables']['categ'], 
                                                modeloStepAIC_int['Variables']['inter'])
Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)

# ModeloStepwise(BIC) con interacciones

modeloStepBIC_int = lm_stepwise(y_train, x_train, var_cont, var_categ,
                                interacciones, 'BIC')
modeloStepBIC_int['Modelo'].summary()
Rsq(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])
x_test_modeloStepBIC_int = crear_data_modelo(x_test, modeloStepBIC_int['Variables']['cont'], 
                                                modeloStepBIC_int['Variables']['categ'], 
                                                modeloStepBIC_int['Variables']['inter'])
Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)

# Tras el análisis de los resultados del summary() para los distintos modelos
# generados sospechamos de la igualdad entre algunos de ellos. Comprobamos que
# no sean modelos idénticos.

comparar_modelos(modeloStepAIC, modeloBackAIC)
comparar_modelos(modeloStepBIC, modeloBackBIC)
comparar_modelos(modeloStepAIC, modeloForAIC)
comparar_modelos(modeloStepBIC, modeloForBIC)
comparar_modelos(modeloStepAIC, modeloStepBIC)
comparar_modelos(modeloStepAIC_int, modeloStepAIC)

# Concluimos del análisis previo que, sin interacciones, todos los modelos que
# comparten la misma métrica (AIC o BIC) son idénticos, así que solo tenemos
# 4 modelos diferentes.

# Hacemos validación cruzada para comparar los 4 modelos anteriores

results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})

for rep in range(20):

    modelo_stepAIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepAIC['Variables']['cont']
        , modeloStepAIC['Variables']['categ']
    )
    modelo_stepBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC['Variables']['cont']
        , modeloStepBIC['Variables']['categ']
    )
    modelo_stepAIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepAIC_int['Variables']['cont']
        , modeloStepAIC_int['Variables']['categ']
        , modeloStepAIC_int['Variables']['inter']
    )
    modelo_stepBIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_int['Variables']['cont']
        , modeloStepBIC_int['Variables']['categ']
        , modeloStepBIC_int['Variables']['inter']
    )

    results_rep = pd.DataFrame({
        'Rsquared': modelo_stepAIC + modelo_stepBIC + modelo_stepAIC_int + modelo_stepBIC_int
        , 'Resample': ['Rep' + str((rep + 1))]*5*4
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5
    })
    results = pd.concat([results, results_rep], axis = 0)
    
# Boxplot de la validacion cruzada

plt.figure(figsize=(10, 6))
plt.grid(True)
grupo_metrica = results.groupby('Modelo')['Rsquared']
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())
plt.xlabel('Modelo')
plt.ylabel('Rsquared')
plt.show()

#Extraemos los valores numéricos de la representación y los grados de libertad:
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
print(media_r2)
std_r2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2)
num_params = [len(modeloStepAIC['Modelo'].params), len(modeloStepBIC['Modelo'].params),
              len(modeloStepAIC_int['Modelo'].params), len(modeloStepBIC_int['Modelo'].params)]
print(num_params)


#Selección de variables aleatoria

variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizar 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3,
                                                            random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_stepwise(y_train2.astype(int), x_train2, var_cont, var_categ,
                         interacciones, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))


print(variables_seleccionadas)
# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x),
                                              variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula',
                                                                   'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia',
                                          ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(frec_ordenada['Formula'][1])]

## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC['Variables']['cont']
        , modeloStepBIC['Variables']['categ']
        , modeloStepBIC['Variables']['inter']
    )
    modelo2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
        , var_1['inter']
    )
    modelo3 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
        , var_2['inter']
    )
    
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 
        , 'Resample': ['Rep' + str((rep + 1))]*5*3
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 
    })
    results = pd.concat([results, results_rep], axis = 0)

# Boxplot de la validacion cruzada 
     
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

#Rescatamos los valores numéricos del gráfico y los grados de libertad:
media_r2_v2 = results.groupby('Modelo')['Rsquared'].mean()
print (media_r2_v2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2_v2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2_v2)
# Contar el número de parámetros en cada modelo
num_params_v2 = [len(modeloStepBIC['Modelo'].params), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+'))]

print(num_params_v2)

# Características del modelo ganador

modelEffectSizes(modeloStepBIC, y_train, x_train,
                 modeloStepBIC['Variables']['cont'],
                 modeloStepBIC['Variables']['categ'])
modeloStepBIC['Modelo'].summary()

Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)

################################
# MODELOS DE REGRESIÓN LOGÍSTICA 
################################

# Partimos de los mismos datos depurados (no es necesario volver a llamarlos).
# Hacemos una partición de los datos e indicamos que la variable 'y' será un
# entero.

x_train, x_test, y_train, y_test = train_test_split(datos_input, varObjBin,
                                                    test_size = 0.2,
                                                    random_state = 123456)

y_train, y_test = y_train.astype(int), y_test.astype(int) 

# Métodos de selección clásica de variables

# Modelo Stepwise(AIC) GLM

modeloStepAIC_glm = glm_stepwise(y_train, x_train, var_cont, var_categ, [], 'AIC')
summary_glm(modeloStepAIC_glm['Modelo'], y_train, modeloStepAIC_glm['X'])
pseudoR2(modeloStepAIC_glm['Modelo'], modeloStepAIC_glm['X'], y_train)
x_test_modeloStepAIC_glm = crear_data_modelo(x_test, modeloStepAIC_glm['Variables']['cont'], 
                                                modeloStepAIC_glm['Variables']['categ'], 
                                                modeloStepAIC_glm['Variables']['inter'])
pseudoR2(modeloStepAIC_glm['Modelo'], x_test_modeloStepAIC_glm, y_test)

curva_roc(x_test_modeloStepAIC_glm, y_test, modeloStepAIC_glm)

# ModeloStepwise(BIC) GLM

modeloStepBIC_glm = glm_stepwise(y_train, x_train, var_cont, var_categ, [], 'BIC')
summary_glm(modeloStepBIC_glm['Modelo'], y_train, modeloStepBIC_glm['X'])
pseudoR2(modeloStepBIC_glm['Modelo'], modeloStepBIC_glm['X'], y_train)
x_test_modeloStepBIC_glm = crear_data_modelo(x_test, modeloStepBIC_glm['Variables']['cont'], 
                                                modeloStepBIC_glm['Variables']['categ'], 
                                                modeloStepBIC_glm['Variables']['inter'])
pseudoR2(modeloStepBIC_glm['Modelo'], x_test_modeloStepBIC_glm, y_test)

# ModeloBackward(AIC) GLM

modeloBackAIC_glm = glm_backward(y_train, x_train, var_cont, var_categ, [], 'AIC')
summary_glm(modeloBackAIC_glm['Modelo'], y_train, modeloBackAIC_glm['X'])
pseudoR2(modeloBackAIC_glm['Modelo'], modeloBackAIC_glm['X'], y_train)
x_test_modeloBackAIC_glm = crear_data_modelo(x_test, modeloBackAIC_glm['Variables']['cont'], 
                                                modeloBackAIC_glm['Variables']['categ'], 
                                                modeloBackAIC_glm['Variables']['inter'])
pseudoR2(modeloBackAIC_glm['Modelo'], x_test_modeloBackAIC_glm, y_test)

# ModeloBackward(BIC) GLM

modeloBackBIC_glm = glm_backward(y_train, x_train, var_cont, var_categ, [], 'BIC')
summary_glm(modeloBackBIC_glm['Modelo'], y_train, modeloBackBIC_glm['X'])
pseudoR2(modeloBackBIC_glm['Modelo'], modeloBackBIC_glm['X'], y_train)
x_test_modeloBackBIC_glm = crear_data_modelo(x_test, modeloBackBIC_glm['Variables']['cont'], 
                                                modeloBackBIC_glm['Variables']['categ'], 
                                                modeloBackBIC_glm['Variables']['inter'])
pseudoR2(modeloBackBIC_glm['Modelo'], x_test_modeloBackBIC_glm, y_test)

# ModeloForward(AIC) GLM

modeloForAIC_glm = glm_forward(y_train, x_train, var_cont, var_categ, [], 'AIC')
summary_glm(modeloForAIC_glm['Modelo'], y_train, modeloForAIC_glm['X'])
pseudoR2(modeloForAIC_glm['Modelo'], modeloForAIC_glm['X'], y_train)
x_test_modeloForAIC_glm = crear_data_modelo(x_test, modeloForAIC_glm['Variables']['cont'], 
                                                modeloForAIC_glm['Variables']['categ'], 
                                                modeloForAIC_glm['Variables']['inter'])
pseudoR2(modeloForAIC_glm['Modelo'], x_test_modeloForAIC_glm, y_test)

# ModeloForward(BIC) GLM

modeloForBIC_glm = glm_forward(y_train, x_train, var_cont, var_categ, [], 'BIC')
summary_glm(modeloForBIC_glm['Modelo'], y_train, modeloForBIC_glm['X'])
pseudoR2(modeloForBIC_glm['Modelo'], modeloForBIC_glm['X'], y_train)
x_test_modeloForBIC_glm = crear_data_modelo(x_test, modeloForBIC_glm['Variables']['cont'], 
                                                modeloForBIC_glm['Variables']['categ'], 
                                                modeloForBIC_glm['Variables']['inter'])
pseudoR2(modeloForBIC_glm['Modelo'], x_test_modeloForBIC_glm, y_test)

# Comprobamos si alguno de los modelos obtenidos con diferentes métodos de
# elección de variables resulta en modelos idénticos
 
comparar_modelos(modeloStepAIC_glm,modeloBackAIC_glm)
comparar_modelos(modeloStepBIC_glm,modeloBackBIC_glm)
comparar_modelos(modeloStepAIC_glm,modeloForAIC_glm)
comparar_modelos(modeloStepBIC_glm,modeloForBIC_glm)
comparar_modelos(modeloBackAIC_glm,modeloForAIC_glm)
comparar_modelos(modeloBackBIC_glm,modeloForBIC_glm)
comparar_modelos(modeloStepAIC_glm,modeloStepBIC_glm)

# Del resultado de las cuatro líneas anteriores se deduce que tenemos cuatro
# modelos diferentes (modeloStepAIC_glm, modeloBackAIC_glm
# y lo mismo con métrica BIC)

# Añadimos ahora interacciones. No obstante, no puedo hacer interacciones entre
# todas las variables continuas (como hacía en el modelo de regresión lineal)
# puesto que aparece multicolinealidad (o un número de parámetros muy elevado).
# Seleccionamos solo unas cuantas variables para las interacciones en este caso.

interacciones_glm = list(itertools.combinations(var_cont, 2)) 

# Modelo Stepwise(AIC) GLM con interacciones

modeloStepAIC_glm_int = glm_stepwise(y_train, x_train, var_cont, var_categ,
                                     interacciones_glm, 'AIC')
summary_glm(modeloStepAIC_glm_int['Modelo'], y_train, modeloStepAIC_glm_int['X'])
pseudoR2(modeloStepAIC_glm_int['Modelo'], modeloStepAIC_glm_int['X'], y_train)
x_test_modeloStepAIC_glm_int = crear_data_modelo(x_test, modeloStepAIC_glm_int['Variables']['cont'], 
                                                modeloStepAIC_glm_int['Variables']['categ'], 
                                                modeloStepAIC_glm_int['Variables']['inter'])
pseudoR2(modeloStepAIC_glm_int['Modelo'], x_test_modeloStepAIC_glm_int, y_test)

# ModeloStepwise(BIC) GLM con interacciones

modeloStepBIC_glm_int = glm_stepwise(y_train, x_train, var_cont, var_categ,
                                     interacciones_glm, 'BIC')
summary_glm(modeloStepBIC_glm_int['Modelo'], y_train, modeloStepBIC_glm_int['X'])
pseudoR2(modeloStepBIC_glm_int['Modelo'], modeloStepBIC_glm_int['X'], y_train)
x_test_modeloStepBIC_glm_int = crear_data_modelo(x_test, modeloStepBIC_glm_int['Variables']['cont'], 
                                                modeloStepBIC_glm_int['Variables']['categ'], 
                                                modeloStepBIC_glm_int['Variables']['inter'])
pseudoR2(modeloStepBIC_glm_int['Modelo'], x_test_modeloStepBIC_glm_int, y_test)

comparar_modelos(modeloStepAIC_glm_int, modeloStepBIC_glm_int)

# Comparamos modelos en base al AUC con validación cruzada

results = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_glm(5, x_train, y_train, 
                                       modeloStepAIC_glm['Variables']['cont'], 
                                       modeloStepAIC_glm['Variables']['categ'], 
                                       modeloStepAIC_glm['Variables']['inter'])
    modelo2VC = validacion_cruzada_glm(5, x_train, y_train, 
                                       modeloBackAIC_glm['Variables']['cont'], 
                                       modeloBackAIC_glm['Variables']['categ'], 
                                       modeloBackAIC_glm['Variables']['inter'])
    modelo3VC = validacion_cruzada_glm(5, x_train, y_train,
                                       modeloStepAIC_glm_int['Variables']['cont'], 
                                       modeloStepAIC_glm_int['Variables']['categ'], 
                                       modeloStepAIC_glm_int['Variables']['inter'])
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'AUC': modelo1VC + modelo2VC + modelo3VC
        , 'Resample': ['Rep' + str((rep + 1))]*5*3  # Etiqueta de repetición (5 repeticiones 3 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 # Etiqueta de modelo (3 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de AUC por modelo
grupo_metrica = results.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('AUC')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

 
    
# Calcular la media del AUC por modelo
results.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar del AUC por modelo
results.groupby('Modelo')['AUC'].std()    
# Contar el número de parámetros en cada modelo
num_params = [len(modeloStepAIC_glm['Modelo'].coef_[0]), len(modeloBackAIC_glm['Modelo'].coef_[0]),
              len(modeloStepAIC_glm_int['Modelo'].coef_[0])]
print(num_params)

# Métodos de selección aleatoria de variables

# Inicializamos el diccionario en el formato deseado.
variables_seleccionadas = {
    'Formula': [],
    'Variables': [],
    'FD': []
    }


# Realizar 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3,
                                                            random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = glm_stepwise(y_train2.astype(int), x_train2, var_cont, var_categ,
                         interacciones, 'BIC')
    
    # Almacenar las variables seleccionadas, la fórmula correspondiente y los grados de libertad
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    var_inter = ['*'.join(item) for item in modelo['Variables']['inter']]
    form = sorted(modelo['Variables']['cont']) + sorted(modelo['Variables']['categ']) + sorted(var_inter) 
    variables_seleccionadas['Formula'].append('+'.join(form))
    variables_seleccionadas['FD'].append(len(modelo['Modelo'].coef_[0]))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula',
                                                                   'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia',
                                          ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(frec_ordenada['Formula'][0])]
fd_1 = variables_seleccionadas['FD'][variables_seleccionadas['Formula'].index(frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(frec_ordenada['Formula'][1])]
fd_2 = variables_seleccionadas['FD'][variables_seleccionadas['Formula'].index(frec_ordenada['Formula'][1])]

# Volvemos a aplicar una validación cruzada ahora entre los dos modelos generados
# aleatoriamente y el candidato a ganador de los modelos generados por métodos
# clásicos de selección de variables

results = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_glm(5, x_train, y_train, 
                                       modeloBackAIC_glm['Variables']['cont'], 
                                       modeloBackAIC_glm['Variables']['categ'], 
                                       modeloBackAIC_glm['Variables']['inter'])
    modelo2VC = validacion_cruzada_glm(5, x_train, y_train, var_1['cont'],
                                       var_1['categ'], var_1['inter'])
    modelo3VC = validacion_cruzada_glm(5, x_train, y_train, var_2['cont'],
                                       var_2['categ'], var_2['inter'])
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'AUC': modelo1VC + modelo2VC + modelo3VC
        , 'Resample': ['Rep' + str((rep + 1))]*5*3  # Etiqueta de repetición (5 repeticiones 3 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 # Etiqueta de modelo (3 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de AUC por modelo
grupo_metrica = results.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('AUC')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico

# Resultados numéricos del boxplot
results.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar del AUC por modelo
results.groupby('Modelo')['AUC'].std()    
# Contar el número de parámetros en cada modelo
num_params = [len(modeloBackAIC_glm['Modelo'].coef_[0]), fd_1, fd_2]
print(num_params)

# Selección del modelo ganador y características varias del mismo

# Obtención del mejor punto de corte

# Generamos una rejilla de puntos de corte
posiblesCortes = np.arange(0, 1.01, 0.01).tolist()  # Generamos puntos de corte de 0 a 1 con intervalo de 0.01
rejilla = pd.DataFrame({
    'PtoCorte': [],
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'PosPredValue': [],
    'NegPredValue': []
})  # Creamos un DataFrame para almacenar las métricas para cada punto de corte

for pto_corte in posiblesCortes:  # Iteramos sobre los puntos de corte
    rejilla = pd.concat(
        [rejilla, sensEspCorte(modeloBackAIC_glm['Modelo'], x_test, y_test, pto_corte,
                               modeloBackAIC_glm['Variables']['cont'],
                               modeloBackAIC_glm['Variables']['categ'],
                               modeloBackAIC_glm['Variables']['inter'])],
        axis=0
    )

rejilla['Youden'] = rejilla['Sensitivity'] + rejilla['Specificity'] - 1  # Calculamos el índice de Youden
rejilla.index = list(range(len(rejilla)))  # Reindexamos el DataFrame para que los índices sean consecutivos



plt.plot(rejilla['PtoCorte'], rejilla['Youden'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Youden')
plt.title('Youden')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Accuracy'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

rejilla['PtoCorte'][rejilla['Youden'].idxmax()]
rejilla['PtoCorte'][rejilla['Accuracy'].idxmax()]

# El resultado es 0.27 para youden y 0.62 para Accuracy
# Los comparamos

sensEspCorte(modeloBackAIC_glm['Modelo'], x_test, y_test, 0.27,
             modeloBackAIC_glm['Variables']['cont'],
             modeloBackAIC_glm['Variables']['categ'],
             modeloBackAIC_glm['Variables']['inter'])
sensEspCorte(modeloBackAIC_glm['Modelo'], x_test, y_test, 0.62,
             modeloBackAIC_glm['Variables']['cont'],
             modeloBackAIC_glm['Variables']['categ'],
             modeloBackAIC_glm['Variables']['inter'])


# Características del modelo ganador y comoparación entre train y test
# Vemos los coeficientes del modelo ganador

coeficientes = modeloBackAIC_glm['Modelo'].coef_
nombres_caracteristicas = crear_data_modelo(x_train,modeloBackAIC_glm['Variables']['cont'],
                                                    modeloBackAIC_glm['Variables']['categ'],
                                                    modeloBackAIC_glm['Variables']['inter'] ).columns  
# Imprime los nombres de las características junto con sus coeficientes
for nombre, coef in zip(nombres_caracteristicas, coeficientes[0]):
    print(f"Variable: {nombre}, Coeficiente: {coef}")
    
summary_glm(modeloBackAIC_glm['Modelo'], y_train, modeloBackAIC_glm['X'])

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
pseudoR2(modeloBackAIC_glm['Modelo'], modeloBackAIC_glm['X'], y_train)
pseudoR2(modeloBackAIC_glm['Modelo'], x_test_modeloBackAIC_glm, y_test)
# Es poca la diferencia, por lo que el modelo se puede considerar robusto

# Calculamos la diferencia del Area bajo la curva ROC en train y test
curva_roc(crear_data_modelo(x_train, modeloBackAIC_glm['Variables']['cont'],
                            modeloBackAIC_glm['Variables']['categ'],
                            modeloBackAIC_glm['Variables']['inter']), y_train, modeloBackAIC_glm)
curva_roc(x_test_modeloBackAIC_glm, y_test, modeloBackAIC_glm)

# Calculamos la diferencia de las medidas de calidad entre train y test 
sensEspCorte(modeloBackAIC_glm['Modelo'], x_train, y_train, 0.62, modeloBackAIC_glm['Variables']['cont'],
             modeloBackAIC_glm['Variables']['categ'],
             modeloBackAIC_glm['Variables']['inter'])
sensEspCorte(modeloBackAIC_glm['Modelo'], x_test, y_test, 0.62, modeloBackAIC_glm['Variables']['cont'],
             modeloBackAIC_glm['Variables']['categ'],
             modeloBackAIC_glm['Variables']['inter'])

