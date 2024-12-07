# Importar bibliotecas para manipulación de datos
import pandas as pd
import numpy as np

# Importar bibliotecas para spark
from pyspark.conf import SparkConf  
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import sqrt, sum as spark_sum, col, when, pow, avg, struct, count, lit, isnan, isnull
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer


# Importar bibliotecas spark machine learning
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT

# Configurar pandas para mostrar todas las filas y columnas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Ignorar advertencias
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def FN_CODIFICACION_VARIABLES_CATEGORICA_NUMERICA(spark_df, umbral_cardinalidad=100):
    """
    Normalizar las variables categoricas y numericas, también realiza el calculo de variables con alta cardinalidad y consolidar todo en un vector.

    Args:
        param df : df_segmento_fijo_masivo_adquision en el que se va a realizar la operación.
        umbral_cardinalidad (Interget) : es un umbral para seleccionar las variables con más de 100 categorias.

    return: 
        Dataframe en spark con el vector consolidados las variables numericas y categoricas ya normalizadas, además el resultado de las columnas con alta cardinalidad.
    """

    # Definir columnas numéricas y categóricas
    columnas_numericas = [col_name for col_name, col_type in spark_df.dtypes if col_type != 'string' and col_name != 'VAL_TARGET']
    columnas_categoricas = [col_name for col_name, col_type in spark_df.dtypes if col_type == 'string' and col_name != 'VAL_TARGET']

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