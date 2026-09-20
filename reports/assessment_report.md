# Freight Rate Machine Learning Assessment

## 1. Objective

The objective of this assessment was to develop a machine learning model capable of predicting freight load rates.

The supplied development dataset contained labeled freight loads from January through October 2025. The trained model was then used to generate predictions for the supplied validation dataset covering November and December 2025, as well as the fixed December prediction scenario.

The final deliverables consist of:

- `validation_predictions.csv`
- December prediction output
- Source code and reproducibility instructions
- This assessment report
- The generated December prediction chart
- A short Loom walkthrough

---

## 2. Data

Three datasets were supplied for the assessment:

| Dataset | Rows | Columns | Purpose |
|---|---:|---:|---|
| Development (`train-test.csv`) | 48,000 | 14 | Model development and validation |
| Validation (`validation.csv`) | 12,000 | 13 | Final prediction generation |
| December inputs | 31 | 7 | Fixed December prediction scenario |

The development dataset covered January 1 through October 31, 2025.

The supplied validation dataset covered November 1 through December 31, 2025.

The December inputs contained one fixed Lexington-to-Fort Wayne route for each day from December 1 through December 31, 2025.

---

## 3. Data Exploration and Quality Findings

Initial profiling identified several relevant characteristics of the development data.

The target variable, `posted_rate`, was strongly right-skewed, with a long upper tail.

Distance showed a strong positive relationship with posted rate and was the strongest individual numerical relationship observed during initial correlation analysis.

The development data contained:

- 300 missing weight values
- 292 negative weight values
- 374 missing `market_index` values

No duplicate rows or duplicate load IDs were identified in the development or supplied validation datasets.

The validation dataset also contained cities that were not observed in the development data. Because the validation dataset supplied geographic coordinates directly, the model could still use geographic information for these observations.

The December prediction inputs did not contain geographic coordinates. The required Lexington and Fort Wayne cities were both present in the development data, allowing their coordinates to be reconstructed from verified development-data mappings without external geographic data.

---

## 4. Feature Availability

The initial feature-availability analysis distinguished between features available across all required prediction scenarios and features that were unavailable for the December inputs.

The final feature set consisted of:

- `pickup`
- `delivery`
- `equipment`
- `distance`
- `weight`
- `day_of_week`
- `day_of_year`
- `day_of_year_sin`
- `day_of_year_cos`
- `days_since_start`
- `pickup_lat`
- `pickup_lon`
- `delivery_lat`
- `delivery_lon`

`load_id` was excluded because it is an identifier rather than a predictive feature.

`market_index` and `quote_signal` were excluded from the final model because they were not available in the December prediction inputs and no reconstruction method was supplied.

---

## 5. Validation Strategy

Random train/test splitting was not used because the prediction task is temporal.

Instead, an expanding-window validation strategy was used to simulate predicting future months using information available at the time.

The validation folds were:

| Fold | Training Period | Validation Period |
|---|---|---|
| Fold 1 | January–June 2025 | July 2025 |
| Fold 2 | January–July 2025 | August 2025 |
| Fold 3 | January–August 2025 | September 2025 |
| Fold 4 | January–September 2025 | October 2025 |

For each fold, preprocessing was fitted only on the corresponding training portion.

The primary evaluation metric was Mean Absolute Error (MAE).

Root Mean Squared Error (RMSE) was used as a secondary metric.

The November–December validation dataset was not used for model selection.

---

## 6. Baseline

A median-rate baseline was established before evaluating machine learning models.

The baseline produced:

- Mean MAE: 1138.41
- Mean RMSE: 1550.11

This provided a reference point for determining whether the machine learning models provided meaningful predictive improvement.

---

## 7. Model Experiments

Several controlled experiments were conducted using the same chronological validation framework.

### Candidate 1 — Common Operational Features

Candidate 1 used:

- pickup
- delivery
- equipment
- distance
- weight
- date-derived features

Results:

- Mean MAE: 167.03
- Mean RMSE: 644.21

Candidate 1 substantially outperformed the median baseline.

### Candidate 2 — Operational + Geographic Features

Candidate 2 added pickup and delivery geographic coordinates.

Results:

- Mean MAE: 161.71
- Mean RMSE: 645.48

The MAE improvement was observed across all four folds.

### Candidate 2B — Weight Quality Features

Candidate 2B added:

- `weight_missing`
- `weight_negative`
- `weight_abs`

Results:

- Mean MAE: 166.41
- Mean RMSE: 643.50

Because MAE was the primary selection metric, these additional features were not retained.

The unusual weight observations were therefore not manually removed or corrected.

### Candidate 3 — Extra Trees

Candidate 3 replaced HistGradientBoosting with Extra Trees while retaining the Candidate 2 feature set.

Results:

- Mean MAE: 163.47
- Mean RMSE: 698.07

Candidate 3 performed worse than Candidate 2 across all four chronological folds.

### Candidate 4 — Log-Transformed Target

Candidate 4 trained the HistGradientBoosting model using `log1p(posted_rate)` and transformed predictions back to the original scale.

Results:

- Mean MAE: 126.92
- Mean RMSE: 629.42

The improvement was observed across all four chronological folds.

### Candidate 5 — Absolute Error Loss

Candidate 5 retained the Candidate 2 feature set and used `absolute_error` as the HistGradientBoosting loss.

Results:

- Mean MAE: 117.95
- Mean RMSE: 627.51

Candidate 5 improved upon Candidate 4 on both mean MAE and mean RMSE.

---

## 8. Hyperparameter Refinement

A limited hyperparameter experiment was performed on Candidate 5.

The tested variants changed one modelling characteristic at a time:

- lower learning rate with more iterations
- increased tree complexity
- increased L2 regularization

The results were:

| Variant | Mean MAE | Mean RMSE |
|---|---:|---:|
| Candidate 5 Control | 117.95 | 627.51 |
| Candidate 5A — Lower LR | 117.13 | 627.38 |
| Candidate 5B — More Complex | 117.10 | 627.65 |
| Candidate 5C — More Regularized | **116.85** | 627.39 |

Candidate 5C achieved the lowest mean MAE under the predefined chronological validation framework.

The modelling configuration was therefore frozen at Candidate 5C.

---

## 9. Final Model

The final model was trained using the complete 48,000-row January–October development dataset.

Model:

`HistGradientBoostingRegressor`

Configuration:

- Loss: `absolute_error`
- Learning rate: `0.05`
- Maximum iterations: `300`
- Maximum leaf nodes: `31`
- L2 regularization: `5.0`
- Random state: `42`

The final feature set consisted of the 14 features described in Section 4.

The final model was not retrained or modified after generating the development validation results.

---

## 10. Final Prediction Generation

The frozen model generated predictions for all 12,000 rows in `validation.csv`.

Prediction integrity checks confirmed:

- 12,000 rows
- Exact expected load ID set
- No duplicate load IDs
- Correct column order
- Numeric predictions
- Finite predictions
- Positive predictions
- No missing predictions

The final output was saved as:

`validation_predictions.csv`

---

## 11. December Prediction

The final model was also applied to all 31 December prediction inputs.

The required December route was:

Lexington → Fort Wayne

The December input fields remained fixed except for the date.

Both cities were present in the development data, allowing their coordinates to be reconstructed without external geographic data.

The generated December predictions contained:

- 31 rows
- One prediction for each day from December 1 through December 31, 2025
- No missing predictions
- Finite predictions
- Positive predictions

The output was saved as:

`december_predictions.csv`

---

## 12. Official Scorer Validation

The supplied `score.py` was executed against the generated prediction files.

The scorer reported:

- Validated 12,000 final predictions.
- Validated 31 fixed December predictions.
- Created `scorer_results/candidate_december.png`.

The supplied scorer does not calculate the final validation performance metrics. Those metrics are calculated by Spotter after submission.

Therefore, the development-fold MAE and RMSE reported in this document are internal model-selection metrics and are not presented as the final assessment score.

---

## 13. Reproducibility

The project uses a Python virtual environment and the supplied dependencies.

Install dependencies with:

```bash
python -m pip install -r requirements.txt