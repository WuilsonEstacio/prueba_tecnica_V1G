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
from pyspark.sql.functions import sqrt, sum, col, when, pow, lit, count, isnan, isnull, row_number, expr, udf, avg, struct, lit
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
from pyspark.sql import DataFrame

# Importar bibliotecas para normalización 
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.linalg import Vectors, VectorUDT
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# importar bibliotecas spark modelos 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import DataFrame

# Importar librerias para las metricas del modelo 
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve

# Importar bibliotecas para ejecutar y guardar modelos
import mlflow.spark
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# importar biblioteca diccionario 
import os
import tempfile

# Ignorar advertencias
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------------------------------------------------------------

# Definir una función UDF para extraer el segundo valor de la columna `probability`
def FN_EXTRACT_PROBA(probability):
    """
    Extraer el segundo valor de la columna `probability` y convertirlo a un número

    Args:
        probability (str): Cadena de caracteres que contiene la columna `probability`.
    
    Returns:
        float: El segundo valor de la columna `probability` convertido a un número.
        """
    return float(probability[1])

# Registrar la UDF en PySpark
extract_proba_udf = udf(FN_EXTRACT_PROBA, DoubleType())

# -------------------------------------------------------------------------------------------------------------------------------------------

def FN_EVALUACION_METRICAS(y_true, y_pred, y_proba, dataset_name, model_name):
    """
    Calcular y guardar las métricas, matriz de confusión, curva ROC y curva Precision-Recall de los diferentes modelos a entrenar.

    Args:
        y_true (list): Lista de valores reales de la variable objetivo.
        y_pred (list): Lista de valores predichos por el modelo.
        y_proba (list): Lista de probabilidades predichas por el modelo.
        dataset_name (str): Nombre del conjunto de datos (e.g., "test", "val").
        model_name (str): Nombre del modelo evaluado.

    Returns: 
        dict: Diccionario con las métricas accuracy, f1_score, y roc_auc.
    """
    # Verificar que todas las listas tengan la misma longitud
    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError(f"Inconsistent input lengths: y_true={len(y_true)}, y_pred={len(y_pred)}, y_proba={len(y_proba)}")
    
    # Convertir las listas a arrays de numpy para facilitar el cálculo
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    # Calcular las métricas de evaluación
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    roc_auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else None

    # Registrar las métricas en MLflow
    mlflow.log_metric(f"{model_name}_{dataset_name}_accuracy", accuracy)
    mlflow.log_metric(f"{model_name}_{dataset_name}_precision", precision)
    mlflow.log_metric(f"{model_name}_{dataset_name}_recall", recall)
    mlflow.log_metric(f"{model_name}_{dataset_name}_f1_score", f1)
    if roc_auc is not None:
        mlflow.log_metric(f"{model_name}_{dataset_name}_roc_auc", roc_auc)

    # Crear la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Verdaderos")
    plt.xlabel("Predichos")
    plt.title(f'Matriz de Confusión ({dataset_name}) - {model_name}')
    
    # Guardar la matriz de confusión en un archivo temporal antes de mostrarla
    temp_file_path = os.path.join(tempfile.gettempdir(), f'confusion_matrix_{dataset_name}_{model_name}.png')
    plt.savefig(temp_file_path)
    plt.close()

    # Registrar la gráfica en MLflow
    mlflow.log_artifact(temp_file_path)

    # Mostrar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Verdaderos")
    plt.xlabel("Predichos")
    plt.title(f'Matriz de Confusión ({dataset_name}) - {model_name}')
    plt.show()

    # Imprimir el reporte de clasificación
    print(f"Modelo: {model_name} {dataset_name} Reporte de Clasificación:\n", classification_report(y_true, y_pred))
    if roc_auc is not None:
        print(f"Área bajo la curva ROC: {roc_auc:.4f}")

    # Calcular y registrar la curva Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    mlflow.log_metric(f"{model_name}_{dataset_name}_pr_auc", pr_auc)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall ({dataset_name}) - {model_name}')
    plt.legend(loc="best")

    temp_pr_file_path = os.path.join(tempfile.gettempdir(), f'precision_recall_curve_{dataset_name}_{model_name}.png')
    plt.savefig(temp_pr_file_path)
    plt.close()

    mlflow.log_artifact(temp_pr_file_path)

    # Calcular y registrar la curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC ({dataset_name}) - {model_name}')
    plt.legend(loc="best")

    temp_roc_file_path = os.path.join(tempfile.gettempdir(), f'roc_curve_{dataset_name}_{model_name}.png')
    plt.savefig(temp_roc_file_path)
    plt.close()

    mlflow.log_artifact(temp_roc_file_path)

    return {"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}

# -------------------------------------------------------------------------------------------------------------------------------------------

def FN_EVALUACION_METRICAS_PREDICCIONES(y_true, y_pred, y_proba, dataset_name, model_name):
    """
    Calcular y mostrar las métricas, matriz de confusión y curvas ROC y Precision-Recall de los modelos.

    Args:
        y_true (Int): Son los registros reales de las personas que realizaron una adquisición del servicio (1) o no (0).
        y_pred (Int): Son las predicciones de los modelos de las personas que realizaron una adquisición (1) o no (0).
        y_proba (float): Es la probabilidad de que un cliente pueda hacer una adquisición o no.
        dataset_name (str): Es el nombre del dataset, en este caso prueba y validación para las métricas.
        model_name (str): Es el nombre del modelo que se entrena.

    Returns: 
        dict: Genera un diccionario de las métricas accuracy, f1_score y roc_auc
    """
    # Verificar que todas las listas tengan la misma longitud
    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError(f"Inconsistent input lengths: y_true={len(y_true)}, y_pred={len(y_pred)}, y_proba={len(y_proba)}")

    # Calcular los verdaderos positivos (tp), verdaderos negativos (tn), falsos positivos (fp) y falsos negativos (fn)
    tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
    tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
    fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
    fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

    # Calcular las métricas de evaluación
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else None
    precision = tp / (tp + fp) if (tp + fp) != 0 else None
    recall = tp / (tp + fn) if (tp + fn) != 0 else None
    f1 = 2 * (precision * recall) / (precision + recall) if (precision is not None and recall is not None and (precision + recall) != 0) else None
    roc_auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else None

    # Mostrar la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Verdaderos")
    plt.xlabel("Predichos")
    plt.title(f'Matriz de Confusión ({dataset_name}) - {model_name}')
    plt.show()

    # Imprimir el reporte de clasificación
    print(f"Modelo: {model_name} {dataset_name} Reporte de Clasificación:\n", classification_report(y_true, y_pred))
    if roc_auc is not None:
        print(f"Área bajo la curva ROC: {roc_auc:.4f}")

    # Calcular y mostrar la curva Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall ({dataset_name}) - {model_name}')
    plt.legend(loc="best")
    plt.show()

    # Calcular y mostrar la curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC ({dataset_name}) - {model_name}')
    plt.legend(loc="best")
    plt.show()

    return {"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}

# -------------------------------------------------------------------------------------------------------------------------------------------

def FN_EVALUAR_MODELO_ACTUAL(modelo_actual_uri, df_predicccion):
    """
    Evalúa el modelo actual en los conjuntos de validación y prueba.

    Args:
        modelo_actual_uri (string): URI del modelo actual en producción.
        val_data (DataFrame): DataFrame de validación.
        test_data (DataFrame): DataFrame de prueba.

    Returns:
        tuple: Métricas de validación y prueba del modelo actual.
    """
    # Finaliza cualquier ejecución activa
    mlflow.end_run()

    # Cargar el modelo actual desde MLflow
    modelo_actual = mlflow.spark.load_model(modelo_actual_uri)

    # Realizar predicciones en el dataset 
    predictions_actual = modelo_actual.transform(df_predicccion)

    # Evaluar el modelo con todo el dataset
    metrics_actual = FN_EVALUACION_METRICAS_PREDICCIONES(
        y_true=predictions_actual.select('target').rdd.flatMap(lambda x: x).collect(),
        y_pred=predictions_actual.select("prediction").rdd.map(lambda row: row[0]).collect(),
        y_proba=predictions_actual.select("probability").rdd.map(lambda row: row['probability'].toArray()[1]).collect(),
        dataset_name="Predicciones",
        model_name="Modelo_Actual"
    )

    return metrics_actual

# -------------------------------------------------------------------------------------------------------------------------------------------

def FN_COMPARAR_METRICAS_MODELO(run_id, metricas_nuevas):
    """
    Compara las métricas registradas de un modelo en MLflow con las métricas obtenidas con nuevos datos.

    Args:
        run_id (str): ID de la ejecución de MLflow que se desea comparar.
        metricas_nuevas (dict): Métricas obtenidas con los nuevos datos (deben incluir accuracy, f1_score y roc_auc).

    Returns:
        dict: Comparación de las métricas antiguas y nuevas.
        bool: Indica si es necesario entrenar nuevamente o no 
    """
    client = MlflowClient()
    
    # Obtener las métricas registradas del modelo
    metricas_registradas = client.get_run(run_id).data.metrics
    
    # Métricas que queremos comparar
    metricas_interes = ['accuracy', 'f1_score', 'roc_auc']
    
    comparacion = {}
    entrenamiento_necesario = "No se necesita entrenar" 
    
    for key in metricas_interes:
        # Construir el prefijo con el nombre del modelo y la métrica
        prefijo = f"RandomForestClassifier_test_{key}"
        if prefijo in metricas_registradas:
            mejora = metricas_nuevas.get(key) >= metricas_registradas[prefijo]
            comparacion[key] = {
                "antiguo": metricas_registradas[prefijo],
                "predicciones": metricas_nuevas.get(key),
                "mejora": mejora
            }
            if not mejora:
                entrenamiento_necesario = "Se necesita volver a entrenar el modelo "

        else:
            comparacion[key] = {
                "nuevo": metricas_nuevas.get(key),
                "nota": "Métrica no registrada anteriormente."
            }
    
    return comparacion, entrenamiento_necesario

# -------------------------------------------------------------------------------------------------------------------------------------------

def FN_REENTRENAR_MODELOS_ADQUISICION(df, nombre_experimento, user, register_model=True):
    """
    Entrenar y seleccionar el mejor modelo con base a los resultados de las métricas.

    Args:
        df_spark (DataFrame): es el dataframe para realizar el modelo adquisición ya preprocesado y limpio.
        nombre_experimento (string): nombre que va tener el experimento en MLflow.
        register_model (bool): si True, registra el modelo en MLflow; si False, solo valida.

    Returns: 
        El mejor modelo entrenado de acuerdo con sus métricas y su ID.
    """
    # Realizar la división del dataset en entrenamiento, prueba y validación 
    train_data, test_data, val_data = df.randomSplit([0.8, 0.1, 0.1], seed=42)
    
    # Balancear las clases a través de la variable weight que es peso 
    class_counts = train_data.groupBy("target").agg(sum(lit(1)).alias("conteo"))
    total_samples = train_data.count()
    weighted_data = train_data.join(class_counts, "target")
    weighted_data = weighted_data.withColumn("weight", total_samples / (class_counts.count() * col("conteo")))
    train_data_balanceado = weighted_data.select("features", "target", "weight")
    
    # Realizar el proceso de registro del experimento para los modelos 
    nombre_experimento = "/Users/" + user + "/" + nombre_experimento
    mlflow.set_experiment(nombre_experimento)

    # Configuración de MLflow con Databricks 
    mlflow.set_registry_uri("databricks-uc")

    # Modelos a entrenar 
    models = [
        LogisticRegression(featuresCol="features", labelCol="target", weightCol="weight"),
        DecisionTreeClassifier(featuresCol="features", labelCol="target", weightCol="weight"),
        RandomForestClassifier(featuresCol="features", labelCol="target", weightCol="weight"),
    ]

    # Establecer los hiperparámetros de los diferentes modelos a entrenar 
    param_grids = []
    for model in models:
        if isinstance(model, LogisticRegression):
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxIter, [10, 20, 30]) \
                .addGrid(model.regParam, [0.1, 0.01, 0.001]) \
                .build()
        elif isinstance(model, DecisionTreeClassifier):
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxDepth, [5, 10, 20]) \
                .addGrid(model.maxBins, [70, 100, 150]) \
                .build()
        elif isinstance(model, RandomForestClassifier):
            param_grid = ParamGridBuilder() \
                .addGrid(model.maxDepth, [5, 10, 20]) \
                .addGrid(model.numTrees, [10, 20, 30]) \
                .addGrid(model.maxBins, [70, 100, 150]) \
                .build()

        param_grids.append((model, param_grid))
    
    best_model = None
    best_model_metrics = None
    best_model_name = None
    best_run_id = None
    best_signature = None

    for model, param_grid in param_grids:
        with mlflow.start_run() as run:
            # Evaluador para medir la precisión del modelo
            evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")

            # Configuración del validador cruzado
            crossval = CrossValidator(estimator=model,
                                      estimatorParamMaps=param_grid,
                                      evaluator=evaluator,
                                      numFolds=3)
            
            # Ajustar el modelo con los datos balanceados
            cv_model = crossval.fit(train_data_balanceado)

            # Loggear los hiperparámetros específicos del modelo
            model_name = model.__class__.__name__
            if isinstance(model, LogisticRegression):
                mlflow.log_param("model_class", model_name)
                mlflow.log_param("max_iter", cv_model.bestModel._java_obj.getMaxIter())
                mlflow.log_param("reg_param", cv_model.bestModel._java_obj.getRegParam())
            elif isinstance(model, DecisionTreeClassifier):
                mlflow.log_param("model_class", model_name)
                mlflow.log_param("max_depth", cv_model.bestModel._java_obj.getMaxDepth())
                mlflow.log_param("max_bins", cv_model.bestModel._java_obj.getMaxBins())
            elif isinstance(model, RandomForestClassifier):
                mlflow.log_param("model_class", model_name)
                mlflow.log_param("max_depth", cv_model.bestModel._java_obj.getMaxDepth())
                mlflow.log_param("num_trees", cv_model.bestModel._java_obj.getNumTrees())
                mlflow.log_param("max_bins", cv_model.bestModel._java_obj.getMaxBins())
            
            # Realizar predicciones en los conjuntos de prueba y validación
            test_predictions = cv_model.transform(test_data)
            val_predictions = cv_model.transform(val_data)
            
            train_data_pandas = train_data_balanceado.toPandas()
            train_predictions_pandas = cv_model.transform(train_data_balanceado).toPandas()
            signature = infer_signature(train_data_pandas, train_predictions_pandas)

            # Evaluar el nuevo modelo
            test_probability_df = test_predictions.withColumn("y_proba", extract_proba_udf(col("probability"))) \
                                                  .select(
                                                      col("target").alias("y_true"),
                                                      col("prediction").alias("y_pred"),
                                                      col("y_proba")
                                                  )

            test_results = test_probability_df.rdd.map(lambda row: (row['y_true'], row['y_pred'], row['y_proba'])).collect()
            y_true_test, y_pred_test, y_proba_test = zip(*test_results)

            test_metrics = FN_EVALUACION_METRICAS(
                y_true=list(y_true_test),
                y_pred=list(y_pred_test),
                y_proba=list(y_proba_test),
                dataset_name="test",
                model_name=model.__class__.__name__
            )
            
            val_probability_df = val_predictions.withColumn("y_proba", extract_proba_udf(col("probability"))) \
                                                .select(
                                                    col("target").alias("y_true"),
                                                    col("prediction").alias("y_pred"),
                                                    col("y_proba")
                                                )

            val_results = val_probability_df.rdd.map(lambda row: (row['y_true'], row['y_pred'], row['y_proba'])).collect()
            y_true_val, y_pred_val, y_proba_val = zip(*val_results)

            val_metrics = FN_EVALUACION_METRICAS(
                y_true=list(y_true_val),
                y_pred=list(y_pred_val),
                y_proba=list(y_proba_val),
                dataset_name="val",
                model_name=model.__class__.__name__
            )
            
            # Registrar el modelo en MLflow si register_model es True
            if register_model:
                mlflow.spark.log_model(spark_model=cv_model.bestModel, artifact_path="spark-model", registered_model_name=f"MD_01_MODELO_{model_name.lower()}_ADQUISICION", signature=signature)
                mlflow.log_param("model_class", model_name)
            
            run_id = run.info.run_id
            
            # Seleccionar el mejor modelo de acuerdo con las métricas de accuracy, f1_score y curva roc
            if best_model_metrics is None or (test_metrics['accuracy'] > best_model_metrics['accuracy'] and test_metrics['f1_score'] > best_model_metrics['f1_score'] and test_metrics['roc_auc'] > best_model_metrics['roc_auc']):
                best_model = cv_model.bestModel
                best_model_metrics = test_metrics
                best_model_name = f"MD_01_MODELO_{model_name.lower()}_ADQUISICION"
                best_run_id = run_id
                best_signature = signature

    # Guardar el mejor modelo en el catálogo de Databricks si register_model es True
    if register_model:
        model_uri = f"runs:/{best_run_id}/spark-model"
        model_name = f"modelos_ciencia_datos.default.{best_model_name}"
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        
        # Realizar versionamiento de los modelos y poner alias al mejor
        client = MlflowClient()
        alias = f"CHAMPION_{best_model_name.split('.')[-1]}"
        client.set_registered_model_alias(model_name, alias, model_version.version)
    
    return best_model, best_run_id
