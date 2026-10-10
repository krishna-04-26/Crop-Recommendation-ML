# Crop Recommendation Under Changing Soil Conditions

An end-to-end Decision Tree classification project for the PCCST 503 Machine Learning Assignment 1 topic: **Crop Recommendation under Changing Soil Conditions**.

The model predicts one of 22 crop labels from soil nutrient levels (N, P, K), temperature, humidity, pH, and rainfall. It is an educational model trained on the supplied dataset; it is not agronomic advice and must not be used for real farm decisions without local expert validation.

## What this project delivers

- Data-quality checks and a reproducible stratified train/validation/test split.
- Validation-based Decision Tree selection using macro F1, accuracy, precision, and recall.
- A final, single held-out-test evaluation with a confusion matrix, per-class metrics, ROC-AUC, error analysis, and feature importance.
- Assignment-specific sensitivity analysis for three representative test samples: one environmental variable is changed at a time by +/-20% while other inputs stay fixed.
- Extracted Decision Tree rules and perturbed-input robustness checks.
- A saved model bundle and a command-line prediction demo.

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the complete experiment

```bash
python -m crop_recommendation.train --data Crop_recommendation.csv
```

This creates ignored, reproducible outputs in `artifacts/` and `reports/generated/`, including a model bundle, metrics tables, plots, sensitivity table, error analysis, rules, and a concise assignment report.

Run the test suite:

```bash
pytest
```

## Demo prediction

After training, run:

```bash
python -m crop_recommendation.predict \
  --N 20 --P 75 --K 200 --temperature 22 \
  --humidity 92 --ph 6.0 --rainfall 110
```

The command returns the predicted crop, confidence, and top three class probabilities. Inputs outside the observed training ranges are rejected, rather than silently extrapolated.

## Minimal desktop interface

First train the model once, then start the native Python window:

```bash
python -m crop_recommendation.train --data Crop_recommendation.csv
python desktop_app.py
```

Users can enter the seven soil and weather measurements, then view the recommended crop and its top three predicted probabilities. This interface uses the native Qt desktop toolkit through PySide6; it does not start a web server or require a browser.

## Methodology

1. The source CSV is validated for schema, numeric values, missing values, duplicate rows, and class distribution.
2. A fixed-seed, stratified 60/20/20 split creates training, validation, and final test data.
3. Candidate entropy Decision Trees with depths 5, 10, and unlimited depth are fit only to training data.
4. The validation set selects the model by macro F1; accuracy is the secondary criterion and a shallower tree resolves exact ties.
5. The untouched test set is evaluated once after selection. Do not use its results to tune the model.

Decision Trees do not require scaling. All features are numerical and no categorical encoding is required. The project retains all seven supplied predictors because each is a stated agronomic or weather measurement and performs no feature transformation.

## Dataset and limitations

`Crop_recommendation.csv` was supplied in the original project repository: <https://github.com/krishna-04-26/Crop-Recommendation-ML>. The report records this provenance; the original repository does not state the dataset's primary collection source or licence, so those should be confirmed before redistribution.

The data contain 2,200 observations and are balanced across 22 labels. A high score on this dataset does not prove local crop suitability. Important omitted factors include geography, season, soil texture, irrigation, pests, market conditions, and expert agronomic constraints.

## Project structure

```text
src/crop_recommendation/  reproducible training, evaluation, reporting, and CLI prediction
tests/                    data and modelling unit tests
reports/                  assignment-report template and generated outputs
data/                     original split files retained from the source repository
```

The original `sklearn.ipynb` remains as a legacy exploratory notebook. Use the modules under `src/` for the clean implementation and demonstration.
