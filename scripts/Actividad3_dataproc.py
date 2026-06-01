#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actividad 3 — version headless para GCP Dataproc (job submit).
Ejecuta el MISMO pipeline del notebook sobre la base global D completa
(NYC Yellow Taxi 2024, 12 meses, ~41M registros) leyendo desde Cloud Storage.
Imprime las metricas clave como evidencia de ejecucion a escala.

Alumno: Oscar Alberto Ramirez Anaya (A01795438) — TC4034
Uso:
  gcloud dataproc jobs submit pyspark gs://<bucket>/Actividad3_dataproc.py \
     --cluster <DATAPROC_CLUSTER> --region us-central1 -- gs://<bucket>/yellow_2024/
"""
import sys
from functools import reduce
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

BASE = (sys.argv[1] if len(sys.argv) > 1
        else "gs://<GCS_BUCKET>/yellow_2024/")

spark = (SparkSession.builder.appName("Actividad3_Dataproc_A01795438").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("="*70); print("Spark", spark.version, "| base:", BASE); print("="*70)

# Cargador robusto: cada archivo mensual se lee con su esquema inferido y se
# castea a tipos canonicos antes de unir (resuelve la heterogeneidad INT64/DOUBLE
# entre meses que rompe un esquema estricto con mergeSchema=false).
def cargar_mes(path):
    d = spark.read.parquet(path)
    return d.select(
        F.col("tpep_pickup_datetime").cast("timestamp").alias("tpep_pickup_datetime"),
        F.col("tpep_dropoff_datetime").cast("timestamp").alias("tpep_dropoff_datetime"),
        F.col("passenger_count").cast("double").alias("passenger_count"),
        F.col("trip_distance").cast("double").alias("trip_distance"),
        F.col("RatecodeID").cast("double").alias("RatecodeID"),
        F.col("PULocationID").cast("long").alias("PULocationID"),
        F.col("payment_type").cast("long").alias("payment_type"),
        F.col("fare_amount").cast("double").alias("fare_amount"),
        F.col("extra").cast("double").alias("extra"),
        F.col("mta_tax").cast("double").alias("mta_tax"),
        F.col("tip_amount").cast("double").alias("tip_amount"),
        F.col("tolls_amount").cast("double").alias("tolls_amount"),
        F.col("improvement_surcharge").cast("double").alias("improvement_surcharge"),
        F.col("total_amount").cast("double").alias("total_amount"),
        F.col("congestion_surcharge").cast("double").alias("congestion_surcharge"),
    )

meses = [f"{BASE}yellow_tripdata_2024-{m:02d}.parquet" for m in range(1, 13)]
df = reduce(DataFrame.unionByName, [cargar_mes(p) for p in meses]).cache()
TOTAL = df.count()
print(f"[D] Base global (Dataproc, 12 meses): {TOTAL:,} registros | {len(df.columns)} columnas")

# Capa Silver
df_silver = (df.filter(F.col("fare_amount") > 0).filter(F.col("trip_distance") > 0)
             .filter(F.col("trip_distance") < 200)
             .filter(F.col("tpep_pickup_datetime") < F.col("tpep_dropoff_datetime"))
             .filter((F.col("RatecodeID").isNull()) | (F.col("RatecodeID") != 99))
             .filter(F.col("PULocationID").isNotNull()).filter(F.col("payment_type").isNotNull())
             .filter(F.year("tpep_pickup_datetime") == 2024))
N_SILVER = df_silver.count()
print(f"[Silver] {N_SILVER:,} registros ({(1-N_SILVER/TOTAL)*100:.2f}% descartado)")

# Variables de caracterizacion -> 12 estratos
manhattan_ids = [4,12,13,24,41,42,43,45,48,50,68,74,75,79,87,88,90,100,103,104,105,107,113,114,116,
                 120,125,127,128,137,140,141,142,143,144,148,151,152,153,158,161,162,163,164,166,170,
                 186,194,202,209,211,224,229,230,231,232,233,234,236,237,238,239,243,244,246,249,261,262,263]
aeropuerto_ids = [1, 132, 138]
df_part = (df_silver
    .withColumn("tipo_dia", F.when(F.dayofweek("tpep_pickup_datetime").isin([1,7]), "Finde").otherwise("Laborable"))
    .withColumn("zona_origen", F.when(F.col("PULocationID").isin(aeropuerto_ids), "Aeropuerto")
                                .when(F.col("PULocationID").isin(manhattan_ids), "Manhattan").otherwise("Otros"))
    .withColumn("tipo_pago", F.when(F.col("payment_type") == 1, "Tarjeta").otherwise("NoTarjeta")))
df_part = df_part.withColumn("particion_id",
    F.concat_ws("|", F.col("tipo_dia"), F.col("zona_origen"), F.col("tipo_pago"))).cache()
print(f"[Particiones] estratos: {df_part.select('particion_id').distinct().count()}")

# Muestreo estratificado proporcional -> M
N_OBJETIVO = 50_000
frac = max(0.0, min(1.0, N_OBJETIVO / N_SILVER))
fracciones = {r["particion_id"]: frac for r in df_part.select("particion_id").distinct().collect()}
M = df_part.sampleBy("particion_id", fractions=fracciones, seed=42)
# Feature engineering + limpieza
M = (M.withColumn("trip_duration_min",
        (F.col("tpep_dropoff_datetime").cast("long")-F.col("tpep_pickup_datetime").cast("long"))/60.0)
      .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
      .withColumn("dia_semana", F.dayofweek("tpep_pickup_datetime"))
      .fillna({"passenger_count":1,"congestion_surcharge":0.0,"extra":0.0,"mta_tax":0.0,
               "tolls_amount":0.0,"improvement_surcharge":0.0})
      .filter((F.col("trip_duration_min")>0)&(F.col("trip_duration_min")<180))
      .filter(F.col("fare_amount")<500)
      .filter((F.col("passenger_count")>=1)&(F.col("passenger_count")<=6))).cache()
N_M = M.count()
print(f"[M] Muestra estratificada preprocesada: {N_M:,} (~{100*N_M/N_SILVER:.3f}% de D)")

# Train/Test 70/30
train, test = M.randomSplit([0.7, 0.3], seed=42)
print(f"[Split] train={train.count():,} test={test.count():,}")

# ── Supervisado: RandomForest ──
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
num_cols = ["trip_distance","fare_amount","extra","mta_tax","tolls_amount","improvement_surcharge",
            "congestion_surcharge","passenger_count","trip_duration_min","pickup_hour","dia_semana"]
cat_cols = ["zona_origen","tipo_dia"]
stages = [StringIndexer(inputCol="tipo_pago", outputCol="label", handleInvalid="error")]
stages += [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in cat_cols]
stages += [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_ohe") for c in cat_cols]
stages += [VectorAssembler(inputCols=num_cols+[c+"_ohe" for c in cat_cols], outputCol="features")]
stages += [RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=60, maxDepth=8, seed=42)]
model = Pipeline(stages=stages).fit(train)
pred = model.transform(test)
auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)
acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy").evaluate(pred)
f1  = MulticlassClassificationEvaluator(labelCol="label", metricName="f1").evaluate(pred)
print(f"[RandomForest] AUC={auc:.4f}  Accuracy={acc:.4f}  F1={f1:.4f}")

# ── No supervisado: KMeans ──
from pyspark.ml.feature import VectorAssembler as VA, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
clus = ["trip_distance","fare_amount","trip_duration_min","pickup_hour","passenger_count"]
feat = VA(inputCols=clus, outputCol="fr").transform(M)
feat = StandardScaler(inputCol="fr", outputCol="fs", withMean=True, withStd=True).fit(feat).transform(feat).cache()
ev = ClusteringEvaluator(featuresCol="fs", metricName="silhouette", distanceMeasure="squaredEuclidean")
best_k, best_s = None, -2
for k in range(2, 8):
    m = KMeans(featuresCol="fs", k=k, seed=42).fit(feat)
    s = ev.evaluate(m.transform(feat))
    print(f"[KMeans] k={k}  WSSSE={m.summary.trainingCost:,.0f}  silhouette={s:.4f}")
    if s > best_s: best_k, best_s = k, s
print(f"[KMeans] k optimo={best_k}  silhouette={best_s:.4f}")

print("="*70); print("JOB DATAPROC COMPLETADO OK"); print("="*70)
spark.stop()
