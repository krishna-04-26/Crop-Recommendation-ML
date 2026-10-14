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

## Double-click to launch

Install Python 3.10 or later (Python 3.12 is a good choice), clone this repository, and open the project folder in your file manager. Double-click the launcher for your operating system:

| Operating system | File to open |
|---|---|
| macOS | `Start Crop Compass.command` |
| Windows | `Start Crop Compass.bat` |
| Linux | `Start Crop Compass.sh` (choose **Run** or **Run in Terminal** in your file manager) |

The launcher automatically creates a local environment, installs dependencies, prepares the model, and opens the desktop app. First-time setup requires internet and can take a few minutes; keep the setup window open until the app appears. Later launches reuse the environment and model. If dependencies change after pulling an update, the launcher refreshes them automatically.

The desktop app opens with a brief animated Crop Compass welcome: a growing sprout and progress bar transition into the main window.

The launchers are Python app starters, not standalone executables: Python must be installed. On Linux, install your distribution's `python3-venv` package if needed; some file managers require enabling execution of executable text files. Git clones preserve the executable permission on the macOS and Linux launchers; ZIP downloads may require granting that permission in file properties. If macOS blocks a downloaded launcher, use Finder's **Open** action and follow the system prompt.

## Manual setup (optional)

Install Python 3.10 or later and Git. Run all commands from the project folder. A graphical desktop session is required for the interface.

### 1. Download the project

```bash
git clone https://github.com/krishna-04-26/Crop-Recommendation-ML.git
cd Crop-Recommendation-ML
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies and the project package

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The last command installs the modules under `src/` so that both training and the desktop app can import them.

### 4. Train the model and open the app

```bash
python -m crop_recommendation.train --data Crop_recommendation.csv
python desktop_app.py
```

Training is required on a fresh clone because generated model files are excluded from Git. After the first training run, launch the app directly with `python desktop_app.py`.

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

Users can enter the seven soil and weather measurements, then view **only the single best predicted crop**. This interface uses the native Qt desktop toolkit through PySide6.

1. Enter the soil nutrients and pH in the **Soil nutrients** panel.
2. Enter temperature, humidity, and rainfall in **Growing conditions**.
3. Select **Find my crop** to display the predicted crop.
4. Select **Use example values** to load a random real dataset sample each time (without consecutive repeats), or **Reset** to restore the input defaults.

For example, N=20, P=75, K=200, temperature=22, humidity=92, pH=6.0, and rainfall=110 produce `Recommended crop: Apple` with the supplied model workflow. Input controls display and enforce the dataset's observed ranges. Nutrient units and the rainfall measurement period must match the source dataset; their primary documentation is not supplied in this repository.

## How it works

The training script reads the CSV, checks its structure, and splits the records into training, validation, and test sets. Each Decision Tree learns numeric split rules from the training set, such as a humidity or nutrient threshold. Validation scores select the configuration before the final test evaluation.

The selected model is saved in `artifacts/decision_tree_model.joblib`. The desktop app loads this file, passes your seven measurements to the prediction module, and displays the class with the highest predicted probability. The desktop display shows a single crop; the optional CLI also exposes probabilities for analysis.

## Troubleshooting

- **`No module named crop_recommendation`:** activate the virtual environment and run `python -m pip install -e .` from the project folder.
- **Model not found:** run the training command before opening the app.
- **Missing dependencies:** run `python -m pip install -r requirements.txt` in the active environment.
- **Qt platform/display error:** launch the app on a machine with a graphical desktop, rather than in a headless terminal session.
- **Windows environment activation is blocked:** you can use `.venv\Scripts\python.exe` in place of `python` for installation, training, and launching, without activating the environment.

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
desktop_app.py           native Python desktop interface
Crop_recommendation.csv  source dataset
requirements.txt         Python dependencies
pyproject.toml           installable package configuration
src/crop_recommendation/  reproducible training, evaluation, reporting, and CLI prediction
tests/                    data and modelling unit tests
reports/                  assignment-report template and generated outputs
data/                     original split files retained from the source repository
```

The original `sklearn.ipynb` remains as a legacy exploratory notebook. Use the modules under `src/` for the clean implementation and demonstration.
