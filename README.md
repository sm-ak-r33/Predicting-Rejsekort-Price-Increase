# Predicting-Rejsekort-Journeys-From 2023
This is an MLflow experiment tracking project for 4 different models that can be assesed [here](https://dagshub.com/smahasanulkarim/Predicting-Rejsekort-Price-Increase-2023.mlflow/#/compare-runs?runs=[%22b701e3cfda5647ab984e1c73733e0c45%22,%222c785a791b244b35b0c55fc33185d45d%22,%222a7a92eecfbe4f2d9910664a5dd1a21d%22,%22d5ba1442340c49c9badafad18e73ad20%22]&experiments=[%220%22])

![output](https://github.com/user-attachments/assets/87caf981-32dc-48e1-a2e9-048fafe7e889)

The details for feature engineering for the statistical models can be found in the paper [here](https://github.com/sm-ak-r33/Predicting-Rejsekort-Price-Increase-2023/blob/main/rejsekort.pdf)  

# STEPS:
# Git clone the repository

``` bash
Project repo: https://github.com/sm-ak-r33/Herbal-EcoDoc.git
```

# Create the environment from the yml file
``` bash
conda env create -f rejse_environment.yml
```

# Activate the environment from the yml file
``` bash
conda activate rejse
```

# After Ingesting New Data for Pipeline Automation
``` bash
dvc init
```

``` bash
dvc repro
```

``` bash
dvc dag
```
