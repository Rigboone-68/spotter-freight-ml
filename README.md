# Freight Rate Machine Learning Assessment

Machine learning solution for predicting freight load rates from
historical shipment data.

## Overview

This project develops a regression model for freight rate prediction
using the supplied January--October 2025 development dataset.

The modelling process uses chronological expanding-window validation to
reflect the temporal nature of the prediction task. After model
selection, the final model is trained on the complete development
dataset and used to generate predictions for the supplied
November--December validation data and the fixed December prediction
scenario.

## Project Structure

``` text
spotter-freight-ml/
├── data/
│   ├── train-test.csv
│   ├── validation.csv
│   └── december-chart-inputs.csv
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   └── 02_model_development.ipynb
├── reports/
│   └── assessment_report.md
├── scorer_results/
│   └── candidate_december.png
├── src/
│   └── freight_rate_model.py
├── december_predictions.csv
├── validation_predictions.csv
├── requirements.txt
├── score.py
└── README.md
```

## Environment Setup

Python 3.12 was used during development.

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

``` bash
python -m pip install -r requirements.txt
```

## Data

The assessment supplied three datasets:

  ---------------------------------------------------------------------------------------
  Dataset                                       Rows              Columns Purpose
  ----------------------------- -------------------- -------------------- ---------------
  `train-test.csv`                            48,000                   14 Model
                                                                          development and
                                                                          validation

  `validation.csv`                            12,000                   13 Final
                                                                          prediction
                                                                          generation

  `december-chart-inputs.csv`                     31                    7 Fixed December
                                                                          prediction
                                                                          scenario
  ---------------------------------------------------------------------------------------

The development dataset covers January 1 through October 31, 2025.

The supplied validation dataset covers November 1 through December 31,
2025.

The December inputs contain a fixed Lexington-to-Fort Wayne route for
each day from December 1 through December 31, 2025.

## Data Quality Findings

Initial profiling identified several relevant characteristics in the
development data:

-   `posted_rate` is strongly right-skewed with a long upper tail.
-   `distance` has a strong positive relationship with `posted_rate`.
-   300 development observations have missing `weight`.
-   292 development observations have negative `weight`.
-   374 development observations have missing `market_index`.
-   No duplicate rows or duplicate load IDs were identified in the
    development data.
-   The validation dataset contains some cities not observed in the
    development dataset.
-   Validation observations provide geographic coordinates directly,
    allowing geographic features to be used for those observations.
-   The December inputs do not contain geographic coordinates. Lexington
    and Fort Wayne are both present in the development data, allowing
    their coordinates to be reconstructed from verified development-data
    mappings without external geographic data.

Negative and missing weight observations were not manually removed or
corrected. Their treatment was evaluated through the validation
experiments.

## Feature Engineering

The final model uses features that are available for the required
prediction scenarios.

### Categorical Features

-   `pickup`
-   `delivery`
-   `equipment`

### Numerical Features

-   `distance`
-   `weight`
-   `day_of_week`
-   `day_of_year`
-   `day_of_year_sin`
-   `day_of_year_cos`
-   `days_since_start`

### Geographic Features

-   `pickup_lat`
-   `pickup_lon`
-   `delivery_lat`
-   `delivery_lon`

`load_id` was excluded because it is an identifier rather than a
predictive feature.

`market_index` and `quote_signal` were excluded from the final model
because they are not available in the December prediction inputs and no
reconstruction method was supplied.

## Validation Strategy

Random train/test splitting was not used because the prediction task is
temporal.

Instead, an expanding-window validation strategy was used to simulate
predicting a future month using information available up to the
preceding month.

  Fold     Training Period           Validation Period
  -------- ------------------------- -------------------
  Fold 1   January--June 2025        July 2025
  Fold 2   January--July 2025        August 2025
  Fold 3   January--August 2025      September 2025
  Fold 4   January--September 2025   October 2025

For every fold, preprocessing was fitted only on the corresponding
training portion.

The primary model-selection metric was Mean Absolute Error (MAE).

Root Mean Squared Error (RMSE) was used as a secondary metric.

The November--December validation dataset was kept untouched during
model selection.

## Baseline

A median-rate baseline was established before evaluating machine
learning models.

The baseline produced:

-   Mean MAE: 1138.41
-   Mean RMSE: 1550.11

This provided a reference point for evaluating whether machine learning
models provided meaningful predictive improvement.

## Model Experiments

Several controlled experiments were performed using the same
chronological validation framework.

### Candidate 1 --- Common Operational Features

Candidate 1 used:

-   pickup
-   delivery
-   equipment
-   distance
-   weight
-   date-derived features

Results:

-   Mean MAE: 167.03
-   Mean RMSE: 644.21

Candidate 1 substantially outperformed the median baseline.

### Candidate 2 --- Operational + Geographic Features

Candidate 2 added pickup and delivery geographic coordinates.

Results:

-   Mean MAE: 161.71
-   Mean RMSE: 645.48

The MAE improvement was observed across all four chronological folds.

### Candidate 2B --- Weight Quality Features

Candidate 2B added:

-   `weight_missing`
-   `weight_negative`
-   `weight_abs`

Results:

-   Mean MAE: 166.41
-   Mean RMSE: 643.50

Because MAE was the primary selection metric, the additional
weight-quality features were not retained.

### Candidate 3 --- Extra Trees

Candidate 3 replaced HistGradientBoosting with Extra Trees while
retaining the Candidate 2 feature set.

Results:

-   Mean MAE: 163.47
-   Mean RMSE: 698.07

Candidate 3 performed worse than Candidate 2 across all four
chronological folds.

### Candidate 4 --- Log-Transformed Target

Candidate 4 trained the HistGradientBoosting model using
`log1p(posted_rate)` and transformed predictions back to the original
rate scale.

Results:

-   Mean MAE: 126.92
-   Mean RMSE: 629.42

The improvement was observed across all four chronological folds.

### Candidate 5 --- Absolute Error Loss

Candidate 5 retained the Candidate 2 feature set and used
`absolute_error` as the HistGradientBoosting loss.

Results:

-   Mean MAE: 117.95
-   Mean RMSE: 627.51

Candidate 5 improved upon Candidate 4 on both mean MAE and mean RMSE.

## Hyperparameter Refinement

A limited hyperparameter experiment was performed on Candidate 5.

The following controlled variants were evaluated:

-   Lower learning rate with more iterations
-   Increased model complexity
-   Increased L2 regularization

Results:

  Variant                                 Mean MAE   Mean RMSE
  ----------------------------------- ------------ -----------
  Candidate 5 Control                       117.95      627.51
  Candidate 5A --- Lower LR                 117.13      627.38
  Candidate 5B --- More Complex             117.10      627.65
  Candidate 5C --- More Regularized     **116.85**      627.39

Candidate 5C achieved the lowest mean MAE under the predefined
chronological validation framework.

No further hyperparameter search was performed.

## Final Model

The final selected model is:

`HistGradientBoostingRegressor`

Configuration:

``` text
loss = absolute_error
learning_rate = 0.05
max_iter = 300
max_leaf_nodes = 31
l2_regularization = 5.0
random_state = 42
```

The final model was trained using the complete 48,000-row
January--October development dataset.

The final model uses the 14 features described in the Feature
Engineering section.

## Final Development Performance

The selected Candidate 5C configuration achieved the following mean
performance across the four expanding-window development folds:

``` text
MAE  ≈ 116.85
RMSE ≈ 627.39
```

These are internal development/model-selection metrics.

They are not the final Spotter assessment score.

## Final Validation Predictions

The frozen final model generated predictions for every row in
`validation.csv`.

Prediction integrity checks confirmed:

-   12,000 rows
-   Exact expected load ID set
-   No duplicate load IDs
-   Correct column order
-   Numeric predictions
-   Finite predictions
-   Positive predictions
-   No missing predictions

The final output was saved as:

`validation_predictions.csv`

The file contains:

``` text
load_id,predicted_rate
```

## December Predictions

The frozen final model was also applied to all 31 December prediction
inputs.

The required December scenario is:

``` text
Pickup: Lexington
Delivery: Fort Wayne
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000 lb
Date: December 1–31, 2025
```

The December inputs do not provide coordinates. Lexington and Fort Wayne
are both present in the development data, so their coordinates were
reconstructed from the verified development-data mappings.

The generated December output contains:

-   31 rows
-   One prediction for each day from December 1 through December 31,
    2025
-   No missing predictions
-   Finite predictions
-   Positive predictions

The output was saved as:

`december_predictions.csv`

The file contains:

``` text
pickup,delivery,distance,equipment,weight,date,predicted_rate
```

## Official Scorer Validation

The supplied `score.py` was executed against the generated prediction
files:

``` bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The supplied scorer successfully reported:

``` text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

The required December prediction chart was generated at:

`scorer_results/candidate_december.png`

The supplied scorer does not calculate the final validation performance
metrics. Those metrics are calculated by Spotter after submission.

Therefore, the development-fold MAE and RMSE reported in this repository
are used only for internal model selection and are not presented as the
final assessment score.

## Reusable Source Implementation

The final modelling implementation is located at:

`src/freight_rate_model.py`

The module provides functions for:

-   Feature engineering
-   Final model construction
-   Final model training
-   Validation prediction generation
-   December coordinate reconstruction
-   December prediction generation

The source implementation was smoke-tested against the supplied datasets
and reproduced the generated validation predictions and December
prediction pipeline.

## Notebooks

The project contains two notebooks documenting the development process.

### `01_data_profiling.ipynb`

Contains:

-   Dataset loading
-   Data integrity checks
-   Numerical feature investigation
-   Suspicious-value analysis
-   Categorical and route coverage
-   Geographic analysis
-   Temporal analysis
-   Target distribution analysis
-   Feature relationships
-   Feature availability
-   Validation strategy

### `02_model_development.ipynb`

Contains:

-   Baseline construction
-   Expanding-window validation
-   Feature engineering
-   Candidate model experiments
-   Weight-quality experiment
-   Model-family comparison
-   Target/loss experiments
-   Hyperparameter refinement
-   Final model selection
-   Final prediction generation
-   Prediction integrity checks
-   Official scorer validation

## Reproducibility

Install dependencies:

``` bash
python -m pip install -r requirements.txt
```

The supplied scorer can be executed with:

``` bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The complete modelling research process is documented in:

-   `notebooks/01_data_profiling.ipynb`
-   `notebooks/02_model_development.ipynb`

The final reusable implementation is located in:

`src/freight_rate_model.py`

The assessment report is located in:

`reports/assessment_report.md`

## Assessment Deliverables

This repository contains:

-   Final source implementation
-   Data profiling notebook
-   Model development notebook
-   Assessment report
-   `validation_predictions.csv`
-   `december_predictions.csv`
-   Official scorer
-   Generated December prediction chart
-   Dependency specification
-   Reproducibility instructions

## Final Note

The model-selection process was based exclusively on the
January--October development data using chronological expanding-window
validation.

The November--December validation data was kept untouched during model
selection. The final model was frozen before generating predictions for
that dataset.

The supplied scorer confirmed that all required prediction files satisfy
the structural and validity requirements. The final assessment
performance on the validation dataset will be determined by Spotter
after submission.
