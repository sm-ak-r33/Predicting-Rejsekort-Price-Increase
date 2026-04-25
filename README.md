# Predicting Rejsekort Journeys — From 2023

An MLflow experiment-tracking project comparing four forecasting models on
Rejsekort passenger journey data.  Results are tracked on
[DagsHub](https://dagshub.com/smahasanulkarim/Predicting-Rejsekort-Price-Increase-2023.mlflow/#/compare-runs?runs=%5B%22b701e3cfda5647ab984e1c73733e0c45%22,%222c785a791b244b35b0c55fc33185d45d%22,%222a7a92eecfbe4f2d9910664a5dd1a21d%22,%22d5ba1442340c49c9badafad18e73ad20%22%5D&experiments=%5B%220%22%5D).

---

## Latest forecast (updated automatically on every run)

![Latest forecast output](output.png)

> This image is regenerated and committed by the
> [nightly GitHub Actions workflow](.github/workflows/nightly.yml) every time
> the pipeline runs, so it always reflects the most recent data from
> [passagertal.dk](https://passagertal.dk).

---

## Feature-engineering details

Full description of statistical features used by the models:
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

### 3 — Install Playwright (required for data ingestion)

```bash
playwright install chromium
```

---

## Running the pipeline

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

## Data ingestion — how it works

Data is sourced from the public TARGIT Anywhere dashboard at
[passagertal.dk](https://passagertal.dk/Embed#vfs://Global/passagertal.dk/Rejsekort/Rejsekortrejser.xview).

Because passagertal.dk does not expose a public REST API, **`pipeline/ingest.py`**
uses [Playwright](https://playwright.dev/python/) (a headless browser) to:

1. Open the TARGIT Anywhere dashboard in a headless Chromium browser.
2. Wait for the dashboard to render its first visualisation.
3. Click the built-in **Export to Excel** button in TARGIT's toolbar.
4. Save the downloaded file as `Data(update).xlsx`, which feeds the rest of
   the DVC pipeline unchanged.

As a fallback, the script intercepts TARGIT's internal XHR data calls and
replays them with an Excel `Accept` header — useful if the export button
cannot be located in a future version of the dashboard.

> **Troubleshooting:** If the export fails, open the dashboard manually in a
> browser, open DevTools → Network, and look for requests to
> `/anywhere/api/data` or `/DataService`.  Update the URL patterns in
> `ingest.py` accordingly.

---

## Pipeline stages (`dvc.yaml`)

| Stage | Script | Input | Output |
|---|---|---|---|
| `ingest` | `pipeline/ingest.py` | passagertal.dk dashboard | `Data(update).xlsx` |
| `script1` | `pipeline/preprocessing.py` | `Data(update).xlsx` | `data_cleaned.csv` |
| `script2` | `pipeline/selected_arima.py` | `data_cleaned.csv` | MLflow run |
| `script3` | `pipeline/autoarima.py` | `data_cleaned.csv` | MLflow run |
| `script4` | `pipeline/prophet_model.py` | `data_cleaned.csv` | MLflow run |
| `script5` | `pipeline/BiLSTM.py` | `data_cleaned.csv` | `output.png` + MLflow run |

---

## Automated nightly refresh (GitHub Actions)

The workflow in `.github/workflows/nightly.yml` runs every night at 04:00 UTC:

1. Installs all dependencies including Playwright + Chromium.
2. Runs `dvc repro --force` to fetch fresh data and re-run all models.
3. Commits the updated `output.png` back to the repository so the plot
   in this README always shows the latest results.

You can also trigger it manually from the **Actions** tab in GitHub.
