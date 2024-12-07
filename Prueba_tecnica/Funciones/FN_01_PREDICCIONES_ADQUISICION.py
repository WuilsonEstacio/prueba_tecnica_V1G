# Importar bibliotecas para manipulación de datos y visualización
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick 
import seaborn as sns

# Importar bibliotecas para manipulación de datos
import pandas as pd
import numpy as np

# Imporat bibliotecas para fechas 
from datetime import datetime

# Importar bibliotecas para Spark
import pyspark.sql.functions as F
from pyspark.conf import SparkConf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import sqrt, sum as spark_sum, col, when, pow, lit, count, isnan, isnull, row_number, expr, udf
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
from pyspark.sql import DataFrame
from pyspark.sql.functions import monotonically_increasing_id

# Importar bibliotecas para normalización 
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT

# Importar bibliotecas para ejecutar mejor modelo 
import mlflow.spark

# Importar bibliotecas para manejo de fechas
import os

# Ignorar advertencias
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------------------------------------------------------------------

def FN_TOMAR_PROBABILIDADES_CLASE_UNO(Probabilities):
    '''
    Extraer las probabilidades de la clase uno del modelo de adquision

    Argmens:
        Probabilities (Vector Dense): Es el vector de las probabilidades de la clasificación uno y cero

    Return: 
        Retorna una lista de probabilidades de clase 1
    '''
    return float(Probabilities[1])

# -----------------------------------------------------------------------------------------------------------------------------------------

def FN_PREDICCIONES(Mejor_modelo_adquisicion, df_segmento_fijo_adquision_prediccion_preprocesado):
    """
    Generar las predicciones del modelo adquision.

    Args:
        Mejor_modelo_adquision (Modelo -MLFlow): es el mejor modelo que se seleccion en el entrenamiento.
        df_segmento_fijo_adquision_prediccion_preprocesado (dataframe -spark): es el dataframe con el cual se van a realizar las predicciones.

    Returns:
        Dataframe - spark con las siguientes columnas "CO_ID_FIJA", "VAL_SCORE_BUYS", "VAL_PREDICCION_ADQUISICION", "DES_BUYS"
    """
    # Hacer predicciones del modelo de adquision 
    predicciones_adquision = Mejor_modelo_adquisicion.transform(df_segmento_fijo_adquision_prediccion_preprocesado)

    # Verificar el número de registros después de la predicción
    print("Número de registros después de la predicción:", predicciones_adquision.count())

    # Cambiar el nombre de la columna 'prediction' a 'VAL_PREDICCION_ADQUISICION'
    predicciones_adquision = predicciones_adquision.withColumnRenamed("prediction", "VAL_PREDICCION_ADQUISICION")

    # Registrar la función como UDF
    TOMAR_PROBABILIDADES_CLASE_UNO_UDF = udf(FN_TOMAR_PROBABILIDADES_CLASE_UNO, DoubleType())

    # Añadir una nueva columna con la probabilidad de la clase 1
    predicciones_adquision = predicciones_adquision.withColumn("VAL_SCORE_BUYS", TOMAR_PROBABILIDADES_CLASE_UNO_UDF(col("probability")))

    # Filtrar solo las predicciones de clase 1
    df_clase1 = predicciones_adquision.filter(col("VAL_PREDICCION_ADQUISICION") == 1)

    # Extraer la probabilidad mínima de clase 1
    min_prob_clase1 = df_clase1.agg({"VAL_SCORE_BUYS": "min"}).collect()[0][0]

    print("La mínima probabilidad cuando el modelo clasifica como clase 1 es:", min_prob_clase1)

    # Establecer etiqueta a las predicciones 
    predicciones_adquision = predicciones_adquision.withColumn("DES_BUYS", when(predicciones_adquision["VAL_PREDICCION_ADQUISICION"]==1.0, 'Buy').otherwise('Do not buy'))

    # Seleccionar solo las variables necesarias para el dataset de predicción 
    predicciones_adquision = predicciones_adquision.select("CO_ID_FIJA", "VAL_SCORE_BUYS", "VAL_PREDICCION_ADQUISICION", "DES_BUYS")

    # Verificar la distribución de la VAL_PREDICCION_ADQUISICION
    predicciones_adquision.groupby("VAL_PREDICCION_ADQUISICION").count().show()

    return predicciones_adquision

# -----------------------------------------------------------------------------------------------------------------------------------------

def FN_CODIFICACION_VARIABLES_CATEGORICA_NUMERICA_EXCLUYENDO_CO_ID_FIJA(spark_df, umbral_cardinalidad=100):
    """
    Normalizar las variables categoricas y numericas, también realiza el calculo de variables con alta cardinalidad y consolidar todo en un vector.

    Args:
        param df : df_segmento_fijo_masivo_adquision en el que se va a realizar la operación.
        umbral_cardinalidad (Integer) : es un umbral para seleccionar las variables con más de 100 categorias.

    return: 
        Dataframe en spark con el vector consolidado de las variables numericas y categoricas ya normalizadas, además el resultado de las columnas con alta cardinalidad.
    """

    # Definir columnas numéricas y categóricas, excluyendo la columna 'CO_ID_FIJA'
    columnas_numericas = [col_name for col_name, col_type in spark_df.dtypes if col_type != 'string' and col_name != 'VAL_TARGET' and col_name != 'CO_ID_FIJA']
    columnas_categoricas = [col_name for col_name, col_type in spark_df.dtypes if col_type == 'string' and col_name != 'VAL_TARGET' and col_name != 'CO_ID_FIJA']

    cardinalidad_alta_encoding = {}
    spark_df_copy = spark_df

    for var in columnas_categoricas:
        cuenta_categorica = spark_df.groupBy(var).agg(count("*").alias("count"))
        if cuenta_categorica.count() > umbral_cardinalidad:
            frecuencia_categorica = cuenta_categorica.withColumn("frequency", col("count") / spark_df.count())
            cardinalidad_alta_encoding[var] = frecuencia_categorica
            spark_df_copy = spark_df_copy.join(frecuencia_categorica, var, "left").withColumn(var + "_encoded", col("frequency")).drop(var).drop("frequency")

    # Obtener las variables categóricas restantes (no de alta cardinalidad)
    columnas_restantes = [col_name for col_name in columnas_categoricas if col_name not in cardinalidad_alta_encoding]

    # Crear una lista de StringIndexer para cada variable categórica restante
    indexar = [StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="skip") for c in columnas_restantes]

    # Transformar las columnas numéricas en una sola columna de vector
    vectorizar_columnas_numericas = VectorAssembler(inputCols=columnas_numericas, outputCol='columnas_numericas')
    escalar = MinMaxScaler(inputCol='columnas_numericas', outputCol='columnas_numericas_escaladas')

    # Transformar todas las columnas de vector creadas en una sola columna de vector
    vectorizar_todas_columnas = [indexador.getOutputCol() for indexador in indexar] + ["columnas_numericas_escaladas"] + [var + "_encoded" for var in cardinalidad_alta_encoding.keys()]
    vectorizar_final = VectorAssembler(inputCols=vectorizar_todas_columnas, outputCol="VAL_FEATURES")

    # Crear el pipeline con los pasos anteriores
    data_pipeline = Pipeline(stages=indexar + [vectorizar_columnas_numericas, escalar, vectorizar_final])

    return data_pipeline, spark_df_copy