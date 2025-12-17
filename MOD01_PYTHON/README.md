# Análisis de Datos de Películas IMDB

Análisis de datos sobre películas de IMDB desarrollado en Python. Incluye exploración de datos, visualizaciones, web scraping y procesamiento con MapReduce.

## Requisitos

- Python 3.7+
- Librerías: `matplotlib`, `requests`, `beautifulsoup4`, `numpy`, `pandas`, `plotly`, `mrjob`

Instalación:
```bash
pip install -r requirements.txt
```

## Datos

**⚠️ IMPORTANTE**: El archivo `movie_data.csv` no está incluido en este repositorio.

Para obtenerlo:
1. Descarga el dataset desde: https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset
2. Coloca el archivo `movie_data.csv` en la misma carpeta que el notebook

## Uso

Ejecuta el notebook `movies_02_enunciados.ipynb` en orden secuencial.

Para la Parte F (MapReduce):
```bash
python language_budget_countries.py -q algunos_campos.txt
```

## Contenido

- **Parte A**: Ejercicios básicos sin pandas
- **Parte B**: Organización de datos en diccionarios
- **Parte C**: Visualizaciones con matplotlib
- **Parte D**: Web scraping de IMDB
- **Parte E**: Análisis con pandas
- **Parte F**: Cálculo masivo con MapReduce
- **Parte G**: Visualización interactiva con mapamundi

## Autor

Jacobo Álvarez Gutiérrez  
Octubre-Noviembre 2024
