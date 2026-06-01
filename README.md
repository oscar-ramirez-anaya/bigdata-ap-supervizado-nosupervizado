<div style="text-align: center;">

# Actividad 3 | Aprendizaje Supervisado y No Supervisado

### Big Data con PySpark — ejecución local y a escala en GCP Dataproc

**Tecnológico de Monterrey** · Maestría en Inteligencia Artificial Aplicada
**TC4034 — Análisis de Grandes Volúmenes de Datos**

**Alumno:** Oscar Alberto Ramírez Anaya · **Matrícula:** A01795438 · **Correo:** A01795438@tec.mx
**Modalidad:** Individual · **Fecha:** Mayo 2026

</div>

---

## Descripción

Este repositorio aplica un algoritmo de **aprendizaje supervisado** (`RandomForestClassifier`) y uno de
**aprendizaje no supervisado** (`KMeans`) con **PySpark (MLlib)** sobre la base global
**D = NYC TLC Yellow Taxi 2024**. Se reutiliza el pipeline de particionamiento (12 estratos por
`tipo_dia × zona_origen × tipo_pago`) y **muestreo estratificado proporcional** desarrollado en la
Evidencia 1 (Módulo 3) para construir una muestra contenida **M**, sobre la que se preprocesa, se hace
la partición train/test y se entrenan los modelos.

El trabajo se entrega en **dos notebooks** que implementan el **mismo pipeline** a dos escalas:

| # | Notebook | Escala | Cómputo | Salidas |
|---|---|---|---|---|
| 1 | **`Actividad3_A01795438.ipynb`** | 3 meses (~9.5M) → M ≈ 50k | PySpark **local** (`local[*]`) | **Ejecutado, con gráficos** |
| 2 | **`Actividad3_A01795438_dataproc.ipynb`** | **12 meses (~41M)** → M ≈ 50k | **GCP Dataproc** (Spark en clúster) | Listo para ejecutar en el clúster |

> **Proyecto 1 (local).** Notebook autocontenido y reproducible en una laptop; descarga los datos
> públicos y ejecuta todo el análisis con visualizaciones seaborn. Es el entregable principal y se
> **visualiza directamente en GitHub** (incluye outputs y gráficos).
>
> **Proyecto 2 (Dataproc).** El mismo análisis llevado a **escala industrial**: lee los 12 meses
> (~41M registros) desde Cloud Storage y se ejecuta en un clúster Dataproc, ya sea de forma interactiva
> (Jupyter del clúster vía Component Gateway) o como *job* PySpark headless
> (`scripts/Actividad3_dataproc.py`). Sostiene la perspectiva de **arquitecto de datos a escala en GCP**.

## Los dos notebooks — descargar o visualizar

**1) Notebook local (ejecutado, con gráficos):**
- 👁️ Visualizar en GitHub: [`Actividad3_A01795438.ipynb`](./Actividad3_A01795438.ipynb)
- 🔎 Visor enriquecido (nbviewer): https://nbviewer.org/github/oscar-ramirez-anaya/bigdata-ap-supervizado-nosupervizado/blob/main/Actividad3_A01795438.ipynb
- ⬇️ Descargar (raw): https://raw.githubusercontent.com/oscar-ramirez-anaya/bigdata-ap-supervizado-nosupervizado/main/Actividad3_A01795438.ipynb

**2) Notebook Dataproc (escala 41M):**
- 👁️ Visualizar en GitHub: [`Actividad3_A01795438_dataproc.ipynb`](./Actividad3_A01795438_dataproc.ipynb)
- ⬇️ Descargar (raw): https://raw.githubusercontent.com/oscar-ramirez-anaya/bigdata-ap-supervizado-nosupervizado/main/Actividad3_A01795438_dataproc.ipynb

## Resultados

**Muestreo (representatividad).** La muestra **M** reproduce las proporciones de los 12 estratos con una
**desviación máxima de 0.0018** respecto a la población.

**Aprendizaje supervisado — RandomForestClassifier** (objetivo `tipo_pago`, binario)

| Métrica | Valor (local, 3 meses) |
|---|---|
| AUC (areaUnderROC) | **0.733** |
| Accuracy | 0.845 |
| F1 | 0.818 |

Por el desbalance (~74% Tarjeta), AUC y F1 son más informativas que la *accuracy*. La tarifa, la
distancia y la duración del viaje, junto con la zona de origen, dominan la importancia de variables.

**Aprendizaje no supervisado — KMeans.** Selección por Silhouette: **k = 2** (Silhouette ≈ **0.770**),
separando trayectos cortos urbanos de traslados largos/aeroportuarios.

**Escala (Dataproc).** El pipeline se ejecutó sobre la **población completa: 41,169,720 registros**
(12 meses) leídos desde `gs://…` en un clúster Dataproc, confirmando que el mismo código escala sin
cambios. Resultados a escala:

| Etapa | Valor (Dataproc, 12 meses / 41M) |
|---|---|
| Base global D | 41,169,720 registros |
| Capa Silver | 39,263,800 (−4.63%) |
| Muestra M | 49,415 |
| Train / Test | 34,630 / 14,785 |
| RandomForest | AUC **0.754** · Accuracy 0.844 · F1 0.820 |
| KMeans | k = 2 · Silhouette **0.765** |

El AUC es ligeramente superior al de la corrida local (0.733) gracias a la mayor representatividad de
la población completa de los 12 meses.

## Mapeo a la rúbrica

1. **Introducción teórica (10%)** — supervisado vs. no supervisado y catálogo de MLlib.
2. **Selección de los datos (20%)** — capa Silver, 12 estratos, muestreo estratificado proporcional → M.
3. **Preparación de los datos (20%)** — nulos, outliers, casts, *feature engineering* y EDA visual.
4. **Train/Test (25%)** — 70/30 reproducible, preservación de clases y control de **fuga de datos**.
5. **Modelos (25%)** — RandomForest (AUC/F1/ROC/importancias) y KMeans (codo + Silhouette + perfiles).

## Cómo ejecutar

**Local (Proyecto 1):**
```bash
bash start_jupyter.sh         # descarga datos si faltan y abre JupyterLab
# Run → Restart Kernel and Run All
```
Entorno: PySpark local (`local[*]`, Java 17). Datos públicos NYC TLC; no se versionan.

**Dataproc (Proyecto 2):**
```bash
bash scripts/00_provision_dataproc.sh    # bucket + staging 12 meses + clúster con Jupyter
# Opción A: abrir el Jupyter del clúster (Component Gateway) y correr el notebook _dataproc
# Opción B: job headless:
#   gcloud dataproc jobs submit pyspark gs://<bucket>/Actividad3_dataproc.py \
#     --cluster <DATAPROC_CLUSTER> --region us-central1 -- gs://<bucket>/yellow_2024/
bash scripts/99_teardown.sh              # destruye el clúster + revoca credenciales
```

### Valores tokenizados (configuración)

> **Nota de seguridad.** Para no exponer infraestructura interna en este repositorio público, los
> identificadores de GCP están **tokenizados**: el proyecto, el bucket y el clúster aparecen como
> los placeholders `<GCP_PROJECT>`, `<GCS_BUCKET>` y `<DATAPROC_CLUSTER>`. Antes de ejecutar los
> scripts de Dataproc, define tus propios valores por variables de entorno:

```bash
export GCP_PROJECT="tu-proyecto-gcp"
export GCS_BUCKET="tu-bucket"
export DATAPROC_CLUSTER="tu-cluster"
```

**Seguridad / gobernanza.** Autenticación por **ADC** (sin llaves JSON); identidad implícita del clúster
para leer GCS; **destrucción** de recursos efímeros y revocación de credenciales al terminar; ningún
secreto ni identificador de infraestructura se versiona (ver `.gitignore` y los valores tokenizados).

## Estructura del repositorio

```
.
├── Actividad3_A01795438.ipynb          # Notebook 1 — local, ejecutado con gráficos (ENTREGABLE)
├── Actividad3_A01795438_dataproc.ipynb # Notebook 2 — escala 41M en Dataproc
├── README.md
├── start_jupyter.sh                    # lanzador local (PySpark + Jupyter)
├── scripts/
│   ├── build_notebook.py               # generador reproducible del notebook local
│   ├── make_dataproc_notebook.py       # genera la variante Dataproc
│   ├── Actividad3_dataproc.py          # job PySpark headless (41M)
│   ├── 00_provision_dataproc.sh        # aprovisionamiento GCP (bucket + staging + clúster)
│   └── 99_teardown.sh                  # destrucción de recursos + revocación de credenciales
└── .gitignore                          # excluye datos, secretos y artefactos
```

---

*Datos: NYC TLC Trip Record Data (dominio público). La base global D y su particionamiento se basan en
el proyecto de la Evidencia 1 (Módulo 3), del cual el autor fue coautor.*
