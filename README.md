# Temporal validation of child stunting prediction models across two Rwanda DHS surveys

Code for the paper on temporal validation of machine learning models for stunting in children aged 0 to 59 months. Models are trained on one Rwanda Demographic and Health Survey and tested, once, on the other:

- **Forward:** train on 2019-20 (n = 3,809), test on 2024-25 (n = 6,715)
- **Reverse:** train on 2024-25, test on 2019-20

Seven models are compared: logistic regression, decision tree, random forest, gradient boosting, XGBoost, LightGBM and CatBoost. Nothing is tuned on the test survey.

## Data (not included)

The survey data are free to registered users from the DHS Program (https://dhsprogram.com). Request the Rwanda 2019-20 and 2024-25 children's recode files. No survey data, per-child predictions or Stata log files are shared here, and `.gitignore` blocks them.

The Python scripts read two cleaned files that you make with `stata/01_clean_both_surveys.do` (see `stata/README.md`) and put in the same folder as the scripts:

| File | Survey |
|---|---|
| `rwanda_stunting_clean_2020_v4.dta` | 2019-20 |
| `rwanda_stunting_clean_2025_b_v4.dta` | 2024-25 |

## Folders

| Folder | Contents |
|---|---|
| `python/` | All analysis scripts (`t0` to `t9`) |
| `results/` | Aggregate text and summary tables from our runs. No per-child data |
| `stata/` | Stata cleaning file that builds the two cleaned survey files |

## File names tell you the direction

Every script that depends on the direction comes as a pair. The name says which survey trains the model and which survey is tested:

- `..._train2020_test2025.py`: **forward**, train on 2019-20, test on 2024-25
- `..._train2025_test2020.py`: **reverse**, train on 2024-25, test on 2019-20

Inside each file, `DIRECTION` is already set to match its name, so you do not edit it. Output files carry the same tag (for example `_train2020_test2025_v7`), so forward and reverse results never overwrite each other. `t4`, `t5` and `t6` are small checks: set `DIRECTION` at the top of each.

## Settings

Besides the direction, the main scripts have a **SETTINGS** block at the top:

| Setting | Values | Meaning |
|---|---|---|
| `HARMONISED` | `False` or `True` | `True` drops the 12 predictors recorded very differently in the two surveys |
| `AGE_MAX` | `None`, `24` or `36` | Age limit in months, applied to both surveys. `None` keeps ages 0 to 59 |

Use the **same settings** in the `t1`, `t2` and `t3` files you run together, so their file names match.

## Run order

Run each direction (forward and reverse) with its own file.

0. `stata/01_clean_both_surveys.do`, once.
1. `t0_codebook_check.py` and `t9_codebook_check.py`: check that the two files use the same labels, units and codes (seconds).
2. `t1_validation_<direction>.py`: the main run (hours).
3. `t2_addons_<direction>.py`: after `t1` of the same direction (minutes).
4. `t3_seed_repeats_<direction>.py`: after `t1` (hours).
5. `t7_review_<direction>.py`: extra analyses (hours). Use `T7_QUICK=1` for a fast test.
6. Optional: `t4`, `t5`, `t6` (logistic regression and harmonisation checks) and `t8` (predictor table).

## Scripts

| Script | What it does |
|---|---|
| `t0_codebook_check.py` | Checks variables, labels and ranges match in both surveys |
| `t1_validation_<direction>.py` | Tuning on the training survey only (4 inner folds grouped by cluster, 100 Optuna trials for each of 7 models, seed 42), F1-optimal threshold from inner out-of-fold predictions, refit, one scoring of the test survey, cluster bootstrap intervals, calibration, SHAP comparison |
| `t2_addons_<direction>.py` | Paired AUC differences with Holm correction, weighted AUC, intercept and Platt recalibration, decision curves, AUC by subgroup. Intervals use a bootstrap within sampling strata |
| `t3_seed_repeats_<direction>.py` | How much the result depends on the tuning seed |
| `t4_lr_grid.py`, `t5_lr_rerun.py` | Logistic regression diagnostics over penalty and C |
| `t6_harmonised_check.py` | Quick check that a harmonised run drops the 12 predictors |
| `t7_review_<direction>.py` | Calibration offset in the training survey, models without class weights, paired differences, transported versus within-survey AUC, recalibration sample size, SHAP of separately trained models |
| `t8_predictor_table.py` | Side-by-side table of all 33 predictors in the two surveys (missing %, mean or category %) |
| `t9_codebook_check.py` | Per-predictor label, type, range and most frequent values in both surveys |

`t4` to `t7` read the data-cleaning code from `t3_seed_repeats_train2020_test2025.py`, so keep that file, with that name, in the same folder.

## Results folder

| File | Content |
|---|---|
| `results/t2_paired_tests_and_primary_both_directions.txt` | Primary performance and 21 paired model comparisons, both directions |
| `results/t7_review_analyses_forward.txt`, `results/t7_review_analyses_reverse.txt` | Parts A to F of the extra analyses |
| `results/predictor_table_by_survey.csv` | The 33 predictors side by side |
| `results/codebook_check.txt` | Output of `t9` |

## Software

Python 3.14; scikit-learn 1.9.0, XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, Optuna 4.9.0, SHAP 0.52.0. Install with `pip install -r requirements.txt`.

## Notes

- The surveys name regions differently ("kigali" and "city of kigali"). The scripts map them to the same labels first.
- Missing values are imputed from the training survey only. A category in the test survey that never appears in training is set to the training mode.
- Random seed 42 is used throughout. Results can differ slightly across package versions.

## Licence and citation

MIT (see `LICENSE`). See `CITATION.cff`. After the Zenodo upload, add the DOI there.
