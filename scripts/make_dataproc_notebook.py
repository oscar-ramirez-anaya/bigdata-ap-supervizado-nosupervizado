#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transforma el notebook local en la version para Dataproc:
- SparkSession del cluster (getOrCreate, YARN) en vez de local[*]
- Lectura robusta desde GCS de los 12 meses (~41M)
- Celda inicial que asegura seaborn
- Limpia outputs para ejecucion interactiva en el cluster
Sale: Actividad3_A01795438_dataproc.ipynb
"""
import nbformat as nbf

nb = nbf.read("Actividad3_A01795438.ipynb", as_version=4)

PIP_CELL = nbf.v4.new_code_cell(
    "# Asegura dependencias de visualizacion en el cluster\n"
    "import importlib, subprocess, sys\n"
    "for pkg in ['seaborn']:\n"
    "    try: importlib.import_module(pkg)\n"
    "    except ImportError: subprocess.run([sys.executable,'-m','pip','install','-q',pkg])\n"
    "print('deps OK')")

SPARK_CELL = (
    "from pyspark.sql import SparkSession\n"
    "from pyspark.sql import functions as F\n"
    "# En Dataproc la sesion se conecta al cluster (YARN); no se usa master local.\n"
    "spark = SparkSession.builder.appName('Actividad3_Dataproc_A01795438').getOrCreate()\n"
    "spark.sparkContext.setLogLevel('ERROR')\n"
    "print('Spark', spark.version, '| modo:', spark.sparkContext.master)")

READ_CELL = (
    "from functools import reduce\n"
    "from pyspark.sql import DataFrame\n"
    "BASE = 'gs://<GCS_BUCKET>/yellow_2024/'\n"
    "\n"
    "# Cargador robusto: cada mes con su esquema inferido + cast a tipos canonicos,\n"
    "# para resolver la heterogeneidad INT64/DOUBLE entre archivos.\n"
    "def cargar_mes(path):\n"
    "    d = spark.read.parquet(path)\n"
    "    return d.select(\n"
    "        F.col('tpep_pickup_datetime').cast('timestamp').alias('tpep_pickup_datetime'),\n"
    "        F.col('tpep_dropoff_datetime').cast('timestamp').alias('tpep_dropoff_datetime'),\n"
    "        F.col('passenger_count').cast('double').alias('passenger_count'),\n"
    "        F.col('trip_distance').cast('double').alias('trip_distance'),\n"
    "        F.col('RatecodeID').cast('double').alias('RatecodeID'),\n"
    "        F.col('PULocationID').cast('long').alias('PULocationID'),\n"
    "        F.col('payment_type').cast('long').alias('payment_type'),\n"
    "        F.col('fare_amount').cast('double').alias('fare_amount'),\n"
    "        F.col('extra').cast('double').alias('extra'),\n"
    "        F.col('mta_tax').cast('double').alias('mta_tax'),\n"
    "        F.col('tip_amount').cast('double').alias('tip_amount'),\n"
    "        F.col('tolls_amount').cast('double').alias('tolls_amount'),\n"
    "        F.col('improvement_surcharge').cast('double').alias('improvement_surcharge'),\n"
    "        F.col('total_amount').cast('double').alias('total_amount'),\n"
    "        F.col('congestion_surcharge').cast('double').alias('congestion_surcharge'),\n"
    "    )\n"
    "\n"
    "meses = [f'{BASE}yellow_tripdata_2024-{m:02d}.parquet' for m in range(1, 13)]\n"
    "df = reduce(DataFrame.unionByName, [cargar_mes(p) for p in meses]).cache()\n"
    "TOTAL = df.count()\n"
    "print(f'Base global D (Dataproc, 12 meses): {TOTAL:,} registros | {len(df.columns)} columnas')")

new_cells = [PIP_CELL]
for c in nb.cells:
    if c.cell_type == "code":
        src = c.source
        if 'master("local[*]")' in src or "master('local[*]')" in src:
            c.source = SPARK_CELL
        elif "SCHEMA = StructType(" in src or "spark.read.schema(SCHEMA)" in src:
            c.source = READ_CELL
        # limpiar outputs para ejecucion fresca en el cluster
        c.outputs = []
        c.execution_count = None
    new_cells.append(c)

nb.cells = new_cells
nbf.write(nb, "Actividad3_A01795438_dataproc.ipynb")
print("Escrito Actividad3_A01795438_dataproc.ipynb con", len(nb.cells), "celdas")
