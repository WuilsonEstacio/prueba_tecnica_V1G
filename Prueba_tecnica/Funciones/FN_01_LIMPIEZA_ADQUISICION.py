# Importar bibliotecas para manipulación de datos y visualización
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick 
import seaborn as sns

# Importar bibliotecas para manipulación de datos
import pandas as pd
import numpy as np

# Importar bibliotecas para Spark
import pyspark
import pyspark.sql.functions as F
from pyspark.conf import SparkConf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import sqrt, sum as spark_sum, col, when, pow, lit, isnan, isnull
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
from pyspark.sql import DataFrame

# Importar bibliotecas para normalización 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Importar bibliotecas para manejo de fechas
import os

# Ignorar advertencias
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------------------------------------------------------

def FN_TRANSFORMAR_TIPOS_DATOS(df: DataFrame, columnas_tipos: dict):
    """
    Transforma los tipos de datos de las columnas especificadas en un DataFrame.

    Args:
        df (DataFrame): El DataFrame de entrada.
        columnas_tipos (dict): Un diccionario donde las claves son los nombres de las columnas 
                               y los valores son los tipos de datos a los que se desea convertir.

    Returns:
        DataFrame: El DataFrame con las columnas transformadas.
    """
    for columna, tipo in columnas_tipos.items():
        df = df.withColumn(columna, col(columna).cast(tipo))
    return df

# ---------------------------------------------------------------------------------------------------------------------------

def FN_IDENTIFICAR_COLUMNAS_NULAS(df: DataFrame):
    """
    Identificar las columnas con valores nulos y NaN, tanto numéricas como categóricas.

    Args:
        df : DataFrame en el que se va a realizar la operación.

    Returns: 
        Tuple: Dos listas, una de columnas numéricas y otra de columnas categóricas con valores nulos o NaN.
    """

    # Identificar columnas con valores nulos o NaN
    columnas_nulas = [
        columna for columna in df.columns 
        if df.select(col(columna)).filter(isnan(col(columna)) | isnull(col(columna))).count() > 0
    ]

    # Columnas numéricas con valores nulos o NaN
    columnas_numericas_nulas = [
        columna for columna in columnas_nulas 
        if 'IntegerType' in str(df.schema[columna].dataType) or 'DoubleType' in str(df.schema[columna].dataType)
    ]

    # Columnas categóricas con valores nulos
    columnas_objetivo_nulas = [
        columna for columna in columnas_nulas 
        if 'IntegerType' not in str(df.schema[columna].dataType) and 'DoubleType' not in str(df.schema[columna].dataType)
    ]

    return columnas_numericas_nulas, columnas_objetivo_nulas

# ---------------------------------------------------------------------------------------------------------------------------

def FN_PLOT_TARGET_DISTRIBUCION(spark_df, target_column="VAL_TARGET"):
    """
    Cuenta la ocurrencia de ceros y unos y calcula el porcentaje de particiación que tiene cada uno en una población total 
    
    Args:
        spark_df: DataFrame en el que se va a realizar la operación (df_limpieza_adquision_mlflow).
        target_column (Interger): Nombre de la columna objetivo (por defecto "VAL_TARGET").

    Returns:
        Retorna un grafico con el porcentaje de particiones de ceros y unos que es la variable objetivo 

    """
    # Agrupar y contar las ocurrencias por la columna objetivo
    churn_counts = spark_df.groupBy(target_column).count().toPandas()
    
    # Establecer los colores para el gráfico de tarta
    colors = sns.color_palette("pastel")
    
    # Configurar la fuente a utilizar
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Crear la figura del gráfico de tarta
    plt.figure(figsize=(4, 4))
    plt.pie(churn_counts["count"], labels=["Adquision","No Adquision"], autopct='%.2f%%', colors=colors)
    plt.title('Distribucion Target')
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

def FN_PLOT_VARIABLE_DISTRIBUCION_POR_TARGET(pd_df, features, target_column="VAL_TARGET"):
    """
    Genera graficos de distribución para cada variables agrupada por la variable objetivo 

    Parámetros:
        pd_df: DataFrame de pandas
        features: Lista de nombres de las características a visualizar
        target_column: Nombre de la columna objetivo

    Returns:
        Retorna varios graficos de distribucion de acuerdo con la variable objetivo 

    """
    # Crear la figura y los ejes para los subplots
    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(8, 6 * len(features)))
    fig.subplots_adjust(hspace=0.4)

    # Si solo hay un subplot, ajustar los ejes
    if len(features) == 1:
        axes = [axes]

    # Crear los gráficos de distribución
    for i, col in enumerate(features):
        sns.histplot(data=pd_df, x=col, hue=target_column, multiple='stack', bins=50, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Distribucion de {col} por {target_column.capitalize()}', fontsize=15)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Frecuencia', fontsize=12)
        axes[i].legend(title=target_column.capitalize(), labels=['Yes', 'No'])

    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

def FN_ENCODE_COLUMNAS_CATEGORICAS(spark_df, columna_excluir = "CO_ID_FIJA"):
    """
    Identificar columnas categóricas en un DataFrame de spark y codificarlas utilizando LabelEncoder.

    Args:
        spark_df: DataFrame de Spark en el que se realizará la operación(df_segmento_fijo_adquision).
        columna_excluir (str): Nombre de la columna que no se escalará.

    Returns:
        DataFrame de pandas con las columnas categóricas codificadas.
    """
    # Obtener las columnas que son de tipo "string" (categoricas), excluyendo columna_excluir
    columnas_objetivos = [col_name for col_name, col_type in spark_df.dtypes if col_type == "string" and col_name != columna_excluir]

    # imprimir las columans categoricas
    print("Variables categóricas en el DataFrame:", columnas_objetivos)

    # Convertir el dataframe de spark a pandas 
    data = spark_df.toPandas()

    # Creae una instancia de LabelEnconder 
    LE = LabelEncoder()

    # Aplicar LabelEncoder a cada columna categorica 
    for col in columnas_objetivos:
        data[col] = LE.fit_transform(data[col])

    print("Todas las variables ahora son numéricas.")

    return data    

# ---------------------------------------------------------------------------------------------------------------------------

def FN_ESCALAR_DATOS(df, columna_excluir = "CO_ID_FIJA"):
    """
    Escala las columnas de un DataFrame, excluyendo la columna especificada.
    
    Args:
        df: DataFrame en el que se va a realizar la operación (df_segmento_fijo_adquision).
        columna_excluir (str): Nombre de la columna que no se escalará.

    Returns:
        Dataframe normalizado excluyendo las columnas_excluir 
    """
    # seleccionar las columnas que vamos a escalar 
    columnas_a_mostrar = [col for col in df.columns if col != columna_excluir]
    
    # Realizar noramlizacion de las columnas numericas 
    escalador = StandardScaler()
    escalador.fit(df[columnas_a_mostrar])
    
    # Generar dataset con las columnas escaladas
    data_escalada = pd.DataFrame(escalador.transform(df[columnas_a_mostrar]), columns=df[columnas_a_mostrar].columns)
    
    print("Todas las variables ahora están escaladas.")
    return data_escalada

# ---------------------------------------------------------------------------------------------------------------------------

def FN_PLOT_VARIABLES_CORRELACION_CON_TARGET(pd_df, target_column="VAL_TARGET"):
    """
    calcula la matriz de correlación, filtra las correlaciones con la columna objetivo y genera un gráfico de barras
    mostrando las correlaciones de las características con el objetivo.

    Args:
        pd_df: DataFrame de pandas (Grafico_df_segmento_fijo_masivo_preprocesado).
        target_column (Interger): Nombre de la columna objetivo "VAL_TARGET".

    Returns:
        Retorna un grafico con la matriz de correlacion, donde muestra las correlaciones de las columnas con la variable objetivo ("VAL_TARGET")

    """
    # Calcular la matriz de correlación
    corr_matrix = pd_df.corr()

    # Filtrar las correlaciones con la columna objetivo
    filtered_corr = corr_matrix[target_column].dropna()
    churn_corr = filtered_corr.sort_values()[:-1]

    # Establecer la paleta de colores
    color_palette = ['#64B3A4', '#FF7F00', '#009E79', '#6495ED']
    
    # Crear la figura del gráfico de barras
    plt.figure(figsize=(25, 25))
    barp = sns.barplot(x=churn_corr.values, y=churn_corr.index, palette=color_palette)
    plt.title('Correlacion variables con variable objetivo', fontsize=20, fontweight='bold')
    plt.xlabel('Correlacion de coeficientes')
    plt.ylabel('Variables')
    plt.grid(True, linestyle='--', alpha=0.6)
    barp.set_yticklabels(barp.get_yticklabels(), fontsize=15, fontweight='bold')
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

def FN_PLOT_MATRIZ_CORRELACION(pd_df, title='Matriz de Correlacion'):
    """
    Genera un mapa de calor de la matriz de correlación.

    Args:
        pd_df: DataFrame de pandas (Grafico_df_segmento_fijo_masivo_preprocesado).
        title (string): Título del gráfico 'Matriz de Correlacion'.

    Returns:
        Grafico de mapa de calor matriz de correlación
    """
    # Calcular la matriz de correlación
    corr_matrix = pd_df.corr()
    
    # Crear el mapa de calor
    plt.figure(figsize=(20, 15))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 6}, fmt=".2f")
    plt.title(title, fontsize=20, fontweight='bold')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=10, fontweight='bold')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=10, fontweight='bold')
    
    # Mostrar el gráfico
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------

def FN_CALCULAR_MODA_Y_REEMPLAZAR_NULOS(df, period_col, group_col, target_col):
    """
    Calcula la moda de target_col agrupada por group_col y reemplaza los valores nulos en target_col con esta moda.

    Args:
        df : df_segmento_fijo_masivo en el que se va a realizar la operación.
        period_col (str) : columna del periodo por la que se va a agrupar.
        group_col (str) : columna de la categoria por la cual se va agrupar.
        target_col (srt) : columna en la que se va a calcular y reemplazar la moda.
    
    return: 
        df_segmento_fijo_masivo con los valores nulos en target_col reemplazados por la moda correspondiente
    """

    # Crear la ventana de partición por period_col y group_col
    window = Window.partitionBy(period_col, group_col)
    
    # Calcular la moda (en este caso, usamos max como aproximación)
    df = df.withColumn(f'mode_{target_col}', F.expr(f'max({target_col})').over(window))
    
    # Reemplazar valores nulos en target_col con la moda calculada
    df = df.withColumn(target_col, F.when(F.col(target_col).isNull(), F.col(f'mode_{target_col}')).otherwise(F.col(target_col)))
    
    # Eliminar la columna temporal mode_target_col
    df = df.drop(f'mode_{target_col}')
    
    return df
