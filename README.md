# Predicting Rejsekort Journeys — From 2023

An MLflow experiment-tracking project comparing four forecasting models on
Rejsekort passenger journey data. Results are tracked on
[DagsHub](https://dagshub.com/smahasanulkarim/Predicting-Rejsekort-Price-Increase-2023.mlflow/#/compare-runs?runs=%5B%22b701e3cfda5647ab984e1c73733e0c45%22,%222c785a791b244b35b0c55fc33185d45d%22,%222a7a92eecfbe4f2d9910664a5dd1a21d%22,%22d5ba1442340c49c9badafad18e73ad20%22%5D&experiments=%5B%220%22%5D).

---

## Latest Forecast

![Latest forecast output](output.png)

> This image is regenerated and committed automatically every time the pipeline
> runs via the [nightly GitHub Actions workflow](.github/workflows/nightly.yml),
> so it always reflects the most recent data from
> [passagertal.dk](https://passagertal.dk).

---

## Feature Engineering

Full description of the statistical features used by the models:
[rejsekort.pdf](rejsekort.pdf)

---

## Setup

### 1 — Clone the repository

```bash
git clone https://github.com/sm-ak-r33/Predicting-Rejsekort-Price-Increase.git
cd Predicting-Rejsekort-Price-Increase
```

### 2 — Create and activate the Conda environment

```bash
conda env create -f rejse_environment.yml
conda activate rejse
```

### 3 — Install the ingestion dependency

```bash
pip install requests
```

---

## Running the Pipeline

### Full automated run (ingest → preprocess → models)

```bash
dvc repro
```

DVC will only re-run stages whose inputs have changed.

### Force a complete re-run

```bash
dvc repro --force
```

### Inspect the pipeline DAG

```bash
dvc dag
```

---

## Data Ingestion

Data is sourced from the public dashboard at
[passagertal.dk](https://passagertal.dk/Embed#vfs://Global/passagertal.dk/Rejsekort/Rejsekortrejser.xview),
which is powered by TARGIT Anywhere (v26.3).

`pipeline/ingest.py` calls the TARGIT `GetModel` endpoint directly:

```
POST https://passagertal.dk/Visual/GetModel
     ?ObjectId=%7BD87FA879-800A-498F-A9F7-0BFFE899D24E%7D
```

The response is JSON containing the full dataset as a base64-encoded Excel file
in `Model.Content`. The script decodes it and saves it as `Data(update).xlsx`,
which feeds the rest of the pipeline unchanged. No browser or Playwright needed —
just `requests`.

Current dataset: **392 rows**, filtered to `<= Previous year (2025)`.

---

## Pipeline Stages

| Stage | Script | Input | Output |
|---|---|---|---|
| `ingest` | `pipeline/ingest.py` | passagertal.dk API | `Data(update).xlsx` |
| `script1` | `pipeline/preprocessing.py` | `Data(update).xlsx` | `data_cleaned.csv` |
| `script2` | `pipeline/selected_arima.py` | `data_cleaned.csv` | MLflow run |
| `script3` | `pipeline/autoarima.py` | `data_cleaned.csv` | MLflow run |
| `script4` | `pipeline/prophet_model.py` | `data_cleaned.csv` | MLflow run |
| `script5` | `pipeline/BiLSTM.py` | `data_cleaned.csv` | `output.png` + MLflow run |

---

## Automated Nightly Refresh

The workflow in `.github/workflows/nightly.yml` runs every night at 04:00 UTC:

1. Installs all dependencies
2. Runs `dvc repro --force` to fetch fresh data and re-run all models
3. Commits the updated `output.png` back to the repository

You can also trigger it manually from the **Actions** tab on GitHub.
