# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:15:43 2025

@author: jacob
"""
################################
################################
# TAREA FINAL MACHINE LEARNING #
################################
################################

# Importamos los módulos y librerías necesarios
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# Lectura de la base de datos
os.chdir('C:/Users/jacob/Desktop/master_ucm/mod8_machine_learning/tema_1/Tarea')
datos = pd.read_excel('datos_tarea25.xlsx')

#########################################################
# APARTADO 1: ANÁLISIS Y DEPURACIÓN DE LA BASE DE DATOS #
#########################################################

# CONSIDERACIONES GENERALES #

# Vemos, para cada variable, la tipología de datos predefinido
datos.dtypes

# Convertimos 'Levy' a entero (Int64 para poder manejar valores missing)
datos['Levy'] = datos['Levy'].replace('-', np.nan).astype('Int64')

# Convertimos 'Engine volume' a float y añadimos una variable binaria para
# especificiar si el motor es turbo
datos['Turbo'] = (datos['Engine volume']
                  .str.contains('Turbo')
                  .map({True: 1, False: 0}))

datos['Engine volume'] = (datos['Engine volume']
                          .str.replace('Turbo', '', regex = True)
                          .astype(float))

# Suprimimos la cadena 'km' de la variable 'Mileage' y convertimos a entero
datos['Mileage'] = (datos['Mileage']
                    .str.replace('km', '', regex = True)
                    .astype(int))

# Convertimos la variable 'Color' en binaria (Int64), siendo 1 blanco y 0 negro
datos['Color'] = datos['Color'].map({'White': 1, 'Black': 0})

# Convertimos la variable 'Leather interior' en binaria, siendo 1 yes y 0 no
datos['Leather interior'] = datos['Leather interior'].map({'Yes': 1, 'No': 0})

# Convertimos la variable 'Wheel' en binaria, siendo 1 left y 0 right
datos['Wheel'] = datos['Wheel'].map({'Left wheel':1, 'Right-hand drive':0})

# Convertimos la variable 'Gear box type' en binaria, siendo:
datos['Gear box type'] = datos['Gear box type'].map({'Automatic':1,
                                                     'Tiptronic':0})

# VALORES ATÍPICOS #

# Análisis descriptivo de las variables numéricas
descriptivos_num = datos.describe().T

# Eliminamos algunos valores atípicos (11 registros en total)
datos = datos[~datos['Engine volume'].isin([0, 0.1, 0.6])].copy()
datos = datos[~datos['Cylinders'].isin([1, 2, 7])].copy()
datos = datos[datos['Mileage'] != 1111111111]

#Histogramas de la variable 'Price'
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(datos['Price'], bins=100, color='steelblue')
plt.title("Distribución original del precio")
plt.xlabel("Precio")
plt.ylabel("Frecuencia")

plt.subplot(1, 2, 2)
plt.hist(np.log1p(datos['Price']), bins=100, color='mediumseagreen')
plt.title("Distribución logarítmica del precio")
plt.xlabel("log(1 + Precio)")
plt.ylabel("Frecuencia")

plt.tight_layout()

# Rango intercuartílico en log(Price)
Q1 = np.log1p(datos['Price']).quantile(0.25)
Q3 = np.log1p(datos['Price']).quantile(0.75)
IQR = Q3 - Q1

lim_inf = np.exp(Q1 - 1.5 * IQR)
lim_sup = np.exp(Q3 + 1.5 * IQR)

fuera_rango = datos[(datos['Price'] < lim_inf) | (datos['Price'] > lim_sup)]
print(f"Porcentaje eliminado: {100 * len(fuera_rango) / len(datos):.3f}%")

#Como son muy pocos valores se eliminarán los registros asociados (7 en total)
datos = datos[(datos['Price'] >= lim_inf) & (datos['Price'] <= lim_sup)].copy()

descriptivos_num = datos.describe().T

# Correcta representación de cada categoría en variables categóricas
datos['Drive wheels'].value_counts(normalize=True)
datos['Fuel type'].value_counts(normalize=True)
datos['Category'].value_counts(normalize=True)
datos['Manufacturer'].value_counts(normalize=True)

# VALORES MISSING #

datos.isnull().sum()

# Solo existen valores missing en la variable 'Levy'. Estudiamos correlaciones
# con las demás variables antes de imputar.

# Crear la columna indicadora de missing
datos['Levy_missing'] = datos['Levy'].isna().astype('int64')

levy_corr = pd.concat([datos.corr(numeric_only=True)['Levy']
                      .sort_values(ascending=False),
                      datos.corr(numeric_only=True)['Levy_missing']
                      .sort_values(ascending=False)],
                      axis=1
    )
levy_corr.columns = ['Levy', 'Levy_missing']
levy_corr = levy_corr.T
levy_corr = levy_corr.drop(columns=['Levy', 'Levy_missing'])


# Imputación de valores nulos
datos_null_imput = datos.copy()
datos_null_imput['Levy'] = datos_null_imput['Levy'].fillna(0)

# Imputación de valores centrales según número de cilindros
medianas_cylinders = datos.groupby('Cylinders')['Levy'].median().round()
datos_central_imput = datos.copy()
datos_central_imput['Levy'] = datos_central_imput['Levy'].fillna(
    datos_central_imput['Cylinders'].map(medianas_cylinders))


# SELECCIÓN DE VARIABLES #

# Separamos variables numéricas y categóricas, así como variables predictoras
# y variable objetivo
y = datos['Color']

cat_cols = datos.select_dtypes(include=['object']).columns.tolist()
num_cols = [col for col in datos.select_dtypes(
    include=['int32', 'int64', 'Int64', 'float64']).columns if col != 'Color']

X_null = datos_null_imput.drop(columns=['Color'])
X_central = datos_central_imput.drop(columns=['Color'])

# Correlación Pearson
def correlacion_pearson(y, X1, X2, num_cols, nombre1='Null',nombre2='Central'):
    corr1 = X1[num_cols].corrwith(y)
    corr2 = X2[num_cols].corrwith(y)

    # Combinar en un DataFrame
    df_corr = pd.DataFrame({
        nombre1: corr1,
        nombre2: corr2
    }).T

    # Visualización tipo heatmap
    plt.figure(figsize=(len(num_cols) * 0.6, 3))
    sns.heatmap(
        df_corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        cbar=True,
        linewidths=0.5
    )
    plt.title('Correlación de Pearson con la variable objetivo')
    plt.xlabel('Variables numéricas')
    plt.ylabel('Método de imputación')
    plt.tight_layout()
    plt.show()

correlacion_pearson(y, X_null, X_central, num_cols)

# V de Cramer
def cramers_v(x, y):
    tabla = pd.crosstab(x, y)
    chi2 = chi2_contingency(tabla, correction=False)[0]
    n = tabla.sum().sum()
    r, k = tabla.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

def grafico_cramer(y, X, cat_cols):
    resultados = {
        col: cramers_v(X[col], y) for col in cat_cols if X[col].nunique() > 1
    }
    resultados_df = pd.DataFrame.from_dict(resultados, orient='index',
                                           columns=['V_de_Cramer'])
    resultados_df = resultados_df.sort_values(by='V_de_Cramer',
                                              ascending=False)

    # Gráfico
    resultados_df.plot(kind='barh', figsize=(8,6), legend=False)
    plt.title('V de Cramér con la variable objetivo')
    plt.xlabel('V de Cramér')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# Basta hacerlo con una sola base de datos porque para estas variables son
# idénticas
grafico_cramer(y, X_null, cat_cols)



# NORMALIZACIÓN Y DUMMIFICACIÓN DE LA BASE DE DATOS #

def transformar_y_guardar(X, y, num_cols, cat_cols, nombre_archivo,
                          nombre_preprocesador):
    # Pipeline de transformación
    preprocesador = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ])

    # Aplicar transformaciones
    X_transformado = preprocesador.fit_transform(X)

    # Obtener nombres de columnas finales
    columnas_cat = (preprocesador.named_transformers_['cat']
                    .get_feature_names_out(cat_cols))
    columnas_finales = num_cols + list(columnas_cat)

    # Crear DataFrame resultante
    df_resultado = pd.DataFrame(X_transformado, columns=columnas_finales,
                                index=X.index)
    df_resultado['Color'] = y

    # Guardar en Excel
    df_resultado.to_excel(nombre_archivo, index=False)
    
    # Guardar el preprocesador
    joblib.dump(preprocesador, nombre_preprocesador)
    

transformar_y_guardar(X_null, y, num_cols, cat_cols, "datos_null_transf.xlsx",
                      "null_scaler.pkl")
transformar_y_guardar(X_central, y, num_cols, cat_cols,
                      "datos_central_transf.xlsx", "central_scaler.pkl")

##########################################################
# APARTADO 2: Modelo con máquina de vector soporte (SVM) #
##########################################################

# Llamada a las bases de datos depuradas en el apartado anterior
os.chdir('C:/Users/jacob/Desktop/master_ucm/mod8_machine_learning/tema_1/Tarea')
data_null = pd.read_excel('datos_null_transf.xlsx')
data_central = pd.read_excel('datos_central_transf.xlsx')

def particion(data):
    y = data['Color']
    X = data.drop(columns=['Color'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y)
    
    return X_train, X_test, y_train, y_test

X_null_train, X_null_test, y_null_train, y_null_test = particion(data_null)
(X_central_train, X_central_test, 
 y_central_train, y_central_test) = particion(data_central)

# DIFERENTES KERNELS #

def kernel_lineal(x, y):
    svc_linear = SVC(kernel='linear', probability=True,
                     max_iter=10000, tol=1e-3, random_state=123)
    
    # Parrilla original: [0.01, 0.1, 1, 10, 100, 1000]
    param_grid_linear = {'C': [14, 16, 18, 20, 22, 24, 26, 28]}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_linear = GridSearchCV(estimator=svc_linear,
                               param_grid=param_grid_linear,
                               scoring=['accuracy', 'roc_auc'],
                               refit='roc_auc', cv=cv,
                               return_train_score=True,
                               n_jobs=1, verbose=2)

    grid_linear.fit(x, y)
    return grid_linear


def kernel_rbf(x, y):
    svc_rbf = SVC(kernel='rbf', probability=True, max_iter=10000, tol=1e-3,
                  random_state=123)
    
    param_grid_rbf = {'C': [0.01, 0.1, 1, 10, 100, 1000],
                      'gamma': [0.01, 0.1, 1, 10, 100, 1000]}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_rbf = GridSearchCV(estimator=svc_rbf, param_grid=param_grid_rbf,
                            scoring=['accuracy', 'roc_auc'], refit='roc_auc',
                            cv=cv, return_train_score=True,
                            n_jobs=1, verbose=2)

    grid_rbf.fit(x, y)
    return grid_rbf

# Kernel lineal
grid_ln_null = kernel_lineal(X_null_train, y_null_train)
print("Lineal (Levy=0):", grid_ln_null.best_params_) # C=24

grid_ln_central = kernel_lineal(X_central_train, y_central_train)
print("Lineal (Levy=Central):", grid_ln_central.best_params_) # C=18

# Kernel rbf
grid_rbf_null = kernel_rbf(X_null_train, y_null_train)
print("RBF (Levy=0):", grid_rbf_null.best_params_)

grid_rbf_central = kernel_rbf(X_central_train, y_central_train)
print("RBF (Levy=Central):", grid_rbf_central.best_params_)

# Resultados gráficos

def graficar_lineal_comparado(grid1, grid2, label1='Imputaciones nulas',
                              label2='Imputaciones centrales'):
    # Extraer resultados
    results1 = grid1.cv_results_
    results2 = grid2.cv_results_

    C_vals1 = (results1['param_C'].data if hasattr(results1['param_C'], 'data')
               else results1['param_C'])
    C_vals2 = (results2['param_C'].data if hasattr(results2['param_C'], 'data')
               else results2['param_C'])

    plt.figure(figsize=(10, 6))

    # Graficar resultados del primer grid (líneas continuas)
    plt.plot(C_vals1, results1['mean_test_accuracy'],
             label=f'Accuracy ({label1})', linestyle='-', color='blue')
    plt.plot(C_vals1, results1['mean_test_roc_auc'], label=f'AUC ({label1})',
             linestyle='-', color='orange')

    # Graficar resultados del segundo grid (líneas discontinuas)
    plt.plot(C_vals2, results2['mean_test_accuracy'],
             label=f'Accuracy ({label2})', linestyle='--', color='blue')
    plt.plot(C_vals2, results2['mean_test_roc_auc'], label=f'AUC ({label2})',
             linestyle='--', color='orange')

    # Personalización
    if min(C_vals1) > 0 and max(C_vals1) / min(C_vals1) > 100:
        plt.xscale('log')
    plt.xlabel('Parámetro C')
    plt.ylabel('Puntuación media (CV)')
    plt.title('Accuracy y AUC vs Parámetro C para kernel lineal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


graficar_lineal_comparado(grid_ln_null, grid_ln_central)

def graficar_heatmaps_comparados(grid_result_1, grid_result_2, metric='roc_auc'):
    if metric not in ['roc_auc', 'accuracy']:
        raise ValueError("La métrica debe ser 'roc_auc' o 'accuracy'.")

    # Preparar los DataFrames
    df1 = pd.DataFrame(grid_result_1.cv_results_)
    df2 = pd.DataFrame(grid_result_2.cv_results_)

    df1['param_C'] = df1['param_C'].astype(float)
    df1['param_gamma'] = df1['param_gamma'].astype(float)
    df2['param_C'] = df2['param_C'].astype(float)
    df2['param_gamma'] = df2['param_gamma'].astype(float)

    pivot1 = df1.pivot(index='param_gamma', columns='param_C',
                       values=f'mean_test_{metric}')
    pivot2 = df2.pivot(index='param_gamma', columns='param_C',
                       values=f'mean_test_{metric}')

    # Establecer escala común
    vmin = min(pivot1.min().min(), pivot2.min().min())
    vmax = max(pivot1.max().max(), pivot2.max().max())

    # Crear figura y ejes
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cbar_ax = fig.add_axes([.91, .3, .02, .4])  
    sns.heatmap(pivot1, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0],
                vmin=vmin, vmax=vmax, cbar=False)
    sns.heatmap(pivot2, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[1],
                vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax)

    # Títulos y etiquetas
    axes[0].set_title(f"{metric.upper()} medio - Imputaciones nulas")
    axes[1].set_title(f"{metric.upper()} medio - Imputaciones Centrales")
    for ax in axes:
        ax.set_xlabel("C")
        ax.set_ylabel("Gamma")

    plt.tight_layout(rect=[0, 0, 0.9, 1])

graficar_heatmaps_comparados(grid_rbf_null, grid_rbf_central,
                             metric='accuracy')
graficar_heatmaps_comparados(grid_rbf_null, grid_rbf_central,
                             metric='roc_auc')

# Definición de modelos óptimos

# Kernel lineal
SVM_ln_null = SVC(kernel='linear', C=20., probability=True, random_state=123)
SVM_ln_central = SVC(kernel='linear', C=20., probability=True,
                     random_state=123)

# Kernel RBF
SVM_rbf_null = SVC(kernel='rbf', C=10., gamma=100., probability=True,
                   random_state=123)
SVM_rbf_central = SVC(kernel='rbf', C=10., gamma=100., probability=True,
                      random_state=123)

modelos = {
    "Lineal - Imputación nula": SVM_ln_null,
    "Lineal - Imputación central": SVM_ln_central,
    "RBF - Imputación nula": SVM_rbf_null,
    "RBF - Imputación central": SVM_rbf_central
}

X_train = {
    "Lineal - Imputación nula": X_null_train,
    "Lineal - Imputación central": X_central_train,
    "RBF - Imputación nula": X_null_train,
    "RBF - Imputación central": X_central_train
}

y_train = {
    "Lineal - Imputación nula": y_null_train,
    "Lineal - Imputación central": y_central_train,
    "RBF - Imputación nula": y_null_train,
    "RBF - Imputación central": y_central_train
}

# Validación cruzada
results_accuracy = {}
results_auc = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

for nombre, modelo in modelos.items():
    X = X_train[nombre]
    y = y_train[nombre]
    
    acc_scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(modelo, X, y, cv=cv, scoring='roc_auc')
    
    results_accuracy[nombre] = acc_scores
    results_auc[nombre] = auc_scores

# Visualización - Accuracy
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(results_accuracy))
plt.title("Validación Cruzada (5 folds) - Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualización - ROC AUC
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(results_auc))
plt.title("Validación Cruzada (5 folds) - ROC AUC")
plt.ylabel("ROC AUC")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Modelo ganador

best_SVM = SVC(kernel='rbf', C=10.0, gamma=100.0, probability=True,
               random_state=123)
best_SVM.fit(X_null_train, y_null_train)
y_pred_SVM = best_SVM.predict(X_null_test)
y_prob_SVM = best_SVM.predict_proba(X_null_test)[:, 1]

# BAGGING #

SVM_bagging = BaggingClassifier(
    estimator = SVC(kernel='rbf', C=10.0, gamma=100.0, probability=True,
                    random_state=123),
    n_estimators=10, max_samples=0.8, bootstrap=True, random_state=123,
    n_jobs=1)
SVM_bagging.fit(X_null_train, y_null_train)
y_pred_bagging = SVM_bagging.predict(X_null_test)
y_prob_bagging = SVM_bagging.predict_proba(X_null_test)[:, 1]

resumen = pd.DataFrame([
    {
        "Modelo": "SVM RBF",
        "Accuracy": round(accuracy_score(y_null_test, y_pred_SVM), 3),
        "AUC": round(roc_auc_score(y_null_test, y_prob_SVM), 3)
    },
    {
        "Modelo": "SVM RBF + Bagging",
        "Accuracy": round(accuracy_score(y_null_test, y_pred_bagging), 3),
        "AUC": round(roc_auc_score(y_null_test, y_prob_bagging), 3)
    }
])

cm_SVM = confusion_matrix(y_null_test, y_pred_SVM)
cm_bagging = confusion_matrix(y_null_test, y_pred_bagging)

# Crear una figura con dos subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot para SVM RBF
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_SVM)
disp1.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title("Matriz de Confusión - SVM RBF")

# Plot para SVM RBF + Bagging
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_bagging)
disp2.plot(ax=axes[1], cmap='Blues', colorbar=False)
axes[1].set_title("Matriz de Confusión - SVM RBF + Bagging")

plt.tight_layout()
plt.show()

###########################################
# APARTADO 3: MODELO CON TÉCNICA STACKING #
###########################################

os.chdir('C:/Users/jacob/Desktop/master_ucm/mod8_machine_learning/tema_1/Tarea')
data_null = pd.read_excel('datos_null_transf.xlsx')

def particion(data):
    y = data['Color']
    X = data.drop(columns=['Color'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y)
    
    return X_train, X_test, y_train, y_test

X_null_train, X_null_test, y_null_train, y_null_test = particion(data_null)

# MODELO BASE 1: REGRESIÓN LOGÍSTICA #

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2'],
    'class_weight': ['balanced']}

lr_model = LogisticRegression(max_iter=10000, random_state=123)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

lr_grid = GridSearchCV(estimator=lr_model, param_grid=lr_param_grid,
                       scoring=['accuracy', 'roc_auc'], refit='roc_auc',
                       cv=cv, return_train_score=True, n_jobs=1, verbose=2)

lr_grid.fit(X_null_train, y_null_train)
print(lr_grid.best_params_)
# {'C': 100, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'lbfgs'}

lr_base = LogisticRegression(**lr_grid.best_params_,
                             max_iter=10000,
                             random_state=123)

# MODELO BASE 2: RANDOM FOREST #

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

rf_model = RandomForestClassifier(random_state=123)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid,
                       scoring=['accuracy', 'roc_auc'], refit='roc_auc',
                       cv=cv, return_train_score=True, n_jobs=1, verbose=2)

rf_grid.fit(X_null_train, y_null_train)
print(rf_grid.best_params_)
# {'class_weight': None, 'max_depth': 15, 'max_features': 'sqrt',
# 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}

rf_base = RandomForestClassifier(**rf_grid.best_params_, random_state=123)

# MODELO BASE 3: XGBOOST #

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [1, 5],
    'scale_pos_weight': [1.158]
    }

xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                          eval_metric='logloss', random_state=123)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid,
                        scoring=['accuracy', 'roc_auc'], refit='roc_auc',
                        cv=cv, return_train_score=True, n_jobs=-1, verbose=2)

xgb_grid.fit(X_null_train, y_null_train)
print(xgb_grid.best_params_)
# {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 7,
# 'n_estimators': 200, 'reg_lambda': 1, 'scale_pos_weight': 1.158,
# 'subsample': 0.8}

xgb_base = XGBClassifier(**xgb_grid.best_params_, objective='binary:logistic',
                         use_label_encoder=False, eval_metric='logloss',
                         random_state=123)

# MODELO BASE 4: SVM CON KERNEL RBF

svm_base = SVC(kernel='rbf', C=10.0, gamma=100.0, probability=True,
               random_state=123)

# COMPARACIÓN MODELOS BASE #

# Validación cruzada de todos los modelos base
modelos = {
    "Regresión logística": lr_base,
    "Random Forest": rf_base,
    "XGBoost": xgb_base,
    "SVM": svm_base
}

results_accuracy = {}
results_auc = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

for nombre, modelo in modelos.items():
    
    acc_scores = cross_val_score(modelo, X_null_train, y_null_train, cv=cv,
                                 scoring='accuracy')
    auc_scores = cross_val_score(modelo, X_null_train, y_null_train, cv=cv,
                                 scoring='roc_auc')
    
    results_accuracy[nombre] = acc_scores
    results_auc[nombre] = auc_scores

# Visualización - Accuracy
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(results_accuracy))
plt.title("Validación Cruzada (5 folds) - Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualización - ROC AUC
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(results_auc))
plt.title("Validación Cruzada (5 folds) - ROC AUC")
plt.ylabel("ROC AUC")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ENSAMBLADO  STACKING #

# Definimos nuevamente los mismos modelos base para que no estén entrenados
estimators = [
    ('lr', LogisticRegression(**lr_grid.best_params_,
                                 max_iter=10000,
                                 random_state=123)),
    ('rf', RandomForestClassifier(**rf_grid.best_params_, random_state=123)),
    ('xgb', XGBClassifier(**xgb_grid.best_params_, objective='binary:logistic',
                             use_label_encoder=False, eval_metric='logloss',
                             random_state=123)),
    ('svm', SVC(kernel='rbf', C=10.0, gamma=100.0, probability=True))
]

meta_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                    max_iter=10000, random_state=123)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

stacking_model = StackingClassifier(estimators=estimators,
                                    final_estimator=meta_model,
                                    cv = cv, stack_method='predict_proba',
                                    passthrough=False, n_jobs=1, verbose=2)

stacking_model.fit(X_null_train, y_null_train)

y_pred = stacking_model.predict(X_null_test)
y_pred_proba = stacking_model.predict_proba(X_null_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_null_test, y_pred)
print(f"Accuracy del stacking: {accuracy:.4f}")

# ROC AUC
roc_auc = roc_auc_score(y_null_test, y_pred_proba)
print(f"ROC AUC del stacking: {roc_auc:.4f}")

# Matriz de Confusión
cm = confusion_matrix(y_null_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')












