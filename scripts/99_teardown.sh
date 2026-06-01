#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  Destruccion de recursos GCP y revocacion de credenciales — Actividad 3
#  TC4034 — Oscar Alberto Ramirez Anaya (A01795438)
#  Ejecutar SIEMPRE al terminar para no incurrir en costos ni dejar credenciales.
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── Valores TOKENIZADOS: configura con variables de entorno ──
#   export GCP_PROJECT="tu-proyecto-gcp"; export GCS_BUCKET="tu-bucket"; export DATAPROC_CLUSTER="tu-cluster"
PROJECT="${GCP_PROJECT:?Define GCP_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
CLUSTER="${DATAPROC_CLUSTER:-act3-cluster}"
BUCKET="${GCS_BUCKET:?Define GCS_BUCKET}"

echo "==> 1. Borrar el cluster Dataproc (corta el computo)"
gcloud dataproc clusters delete "$CLUSTER" --region "$REGION" --project "$PROJECT" --quiet \
  || echo "   (cluster inexistente o ya borrado)"

echo "==> 2. (Opcional) borrar el bucket de datos del experimento"
read -r -p "   ¿Borrar gs://${BUCKET}? [y/N] " resp
if [[ "${resp:-N}" =~ ^[Yy]$ ]]; then
  gcloud storage rm --recursive "gs://${BUCKET}" --project "$PROJECT" || true
fi

echo "==> 3. Revocar credenciales locales efimeras (ADC)"
gcloud auth application-default revoke --quiet || true

echo "==> 4. Verificacion final (no debe quedar cluster)"
gcloud dataproc clusters list --region "$REGION" --project "$PROJECT"
echo "Destruccion completada."
