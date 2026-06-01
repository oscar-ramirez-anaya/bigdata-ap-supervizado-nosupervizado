#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Aprovisionamiento GCP Dataproc — Actividad 3 (escalado de la base global D)
#  TC4034 — Oscar Alberto Ramirez Anaya (A01795438)
#
#  Procesa la base completa NYC Yellow Taxi 2024 (12 parquet, ~41M registros)
#  en un cluster Dataproc efimero con Jupyter. Coste contenido: --max-idle 2h
#  + script de destruccion (99_teardown.sh).
#
#  Seguridad: autenticacion por ADC (Application Default Credentials), SIN
#  llaves JSON. El job de Spark usa la identidad implicita del cluster.
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configura estos valores con variables de entorno (estan TOKENIZADOS en el repo) ──
#   export GCP_PROJECT="tu-proyecto-gcp"
#   export GCS_BUCKET="tu-bucket"
#   export DATAPROC_CLUSTER="tu-cluster"
PROJECT="${GCP_PROJECT:?Define GCP_PROJECT: export GCP_PROJECT=tu-proyecto-gcp}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
BUCKET="${GCS_BUCKET:?Define GCS_BUCKET: export GCS_BUCKET=tu-bucket}"
CLUSTER="${DATAPROC_CLUSTER:-act3-cluster}"
BASE_URL="https://d37ci6vzurychx.cloudfront.net/trip-data"

echo "==> 0. Contexto"
gcloud config set project "$PROJECT"
gcloud config set compute/region "$REGION"
gcloud config set compute/zone "$ZONE"

echo "==> 1. Habilitar APIs (idempotente)"
gcloud services enable dataproc.googleapis.com compute.googleapis.com \
  storage.googleapis.com --project "$PROJECT"

echo "==> 2. Autenticacion segura (ADC, sin llaves JSON)"
gcloud auth application-default print-access-token >/dev/null 2>&1 \
  || gcloud auth application-default login

echo "==> 3. Bucket de datos"
gcloud storage buckets create "gs://${BUCKET}" \
  --location="$REGION" --uniform-bucket-level-access --project "$PROJECT" \
  || echo "   (bucket ya existe)"

echo "==> 4. Stagear los 12 parquet de NYC TLC (cloudfront -> GCS)"
for m in $(seq -w 1 12); do
  curl -fsSL "${BASE_URL}/yellow_tripdata_2024-${m}.parquet" \
    | gcloud storage cp - "gs://${BUCKET}/yellow_2024/yellow_tripdata_2024-${m}.parquet"
  echo "   OK 2024-${m}"
done
gcloud storage ls -l "gs://${BUCKET}/yellow_2024/"

echo "==> 5. Crear cluster Dataproc efimero con Jupyter + Component Gateway"
gcloud dataproc clusters create "$CLUSTER" \
  --region "$REGION" --project "$PROJECT" \
  --image-version 2.2-debian12 \
  --master-machine-type n2-standard-4 --master-boot-disk-size 100 \
  --num-workers 2 --worker-machine-type n2-standard-4 --worker-boot-disk-size 100 \
  --optional-components JUPYTER \
  --enable-component-gateway \
  --max-idle 2h

echo "==> 6. Enlace a JupyterLab (Component Gateway)"
gcloud dataproc clusters describe "$CLUSTER" --region "$REGION" --project "$PROJECT" \
  --format="value(config.endpointConfig.httpPorts)"

echo ""
echo "Listo. Abre el URL de JupyterLab, sube Actividad3_A01795438.ipynb,"
echo "cambia DATA_PATH a gs://${BUCKET}/yellow_2024/yellow_tripdata_2024-*.parquet y Run All."
echo "Al terminar ejecuta: bash scripts/99_teardown.sh"
