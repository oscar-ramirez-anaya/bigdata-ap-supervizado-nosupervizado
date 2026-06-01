#!/bin/bash
# ══════════════════════════════════════════════════════
#  Lanzador de Jupyter con PySpark — TC4034 Actividad 3
#  Alumno: Oscar Alberto Ramirez Anaya (A01795438)
# ══════════════════════════════════════════════════════
export SPARK_HOME=/opt/homebrew/opt/apache-spark/libexec
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.17/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

echo "╔══════════════════════════════════════════╗"
echo "║   PySpark + Jupyter — TC4034.10          ║"
echo "║   Actividad 3 — Oscar Ramirez A01795438  ║"
echo "╚══════════════════════════════════════════╝"
echo ""
python3 -c "import pyspark; print('PySpark', pyspark.__version__, 'listo')" || {
    echo "PySpark no importable. Verifica la instalacion."; exit 1; }

# Asegura que existan los datos (descarga 3 meses si faltan)
if ! ls data/yellow_2024/*.parquet >/dev/null 2>&1; then
  echo "Descargando 3 meses de datos (enero-marzo 2024)..."
  BASE=https://d37ci6vzurychx.cloudfront.net/trip-data
  mkdir -p data/yellow_2024
  for m in 01 02 03; do
    curl -fsSL -o data/yellow_2024/yellow_tripdata_2024-$m.parquet \
      "$BASE/yellow_tripdata_2024-$m.parquet" && echo "  OK 2024-$m"
  done
fi

cd "$(dirname "$0")"
echo ""
echo "Abriendo JupyterLab. Abre Actividad3_A01795438.ipynb y ejecuta todas las celdas."
python3 -m jupyter lab Actividad3_A01795438.ipynb
