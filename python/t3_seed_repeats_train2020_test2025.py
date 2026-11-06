# PRESET: train on 2019-20, test on 2024-25 (forward).
#======================================================================
# t3: TEMPORAL SEED REPEATS (run after t1, with the same SETTINGS)
#
# What it does
#   Repeats the whole temporal procedure with 5 different seeds (new inner folds, new Optuna seed, new model
#   seed; 40 trials per model) to show how much the temporal AUC depends on tuning and model randomness.
#   Each repeat trains on the TRAINING survey only and scores the TEST survey once.
#   Progress is saved after every repeat; if the run stops, start it again and it continues.
#
# Output: summary tables only (mean, SD and range of AUC, calibration and threshold metrics per model).
#======================================================================
import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------------------
# SETTINGS (the only part you normally change)
# ---------------------------------------------------------------------------------------
DIRECTION  = "forward"   # "forward": train on 2019-20, test on 2024-25.  "reverse": the other way round.
HARMONISED = False       # True: drop the 12 predictors recorded very differently in the two surveys
AGE_MAX    = None        # None: children aged 0-59 months.  24: children under 24 months only
# ---------------------------------------------------------------------------------------

# Data and results folder = the folder that holds this script (put the two .dta files there).
# To use another folder, replace the next lines with: OUT_DIR = r"C:\path\to\folder"
try:
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running in a notebook
    OUT_DIR = os.getcwd()
if DIRECTION == "forward":
    TRAIN_FILE = "rwanda_stunting_clean_2020_v4.dta"    # training survey (2019-20)
    TEST_FILE  = "rwanda_stunting_clean_2025_b_v4.dta"  # test survey (2024-25), scored once
    RUN_TAG    = "_train2020_test2025_v7"
elif DIRECTION == "reverse":
    TRAIN_FILE = "rwanda_stunting_clean_2025_b_v4.dta"  # training survey (2024-25)
    TEST_FILE  = "rwanda_stunting_clean_2020_v4.dta"    # test survey (2019-20), scored once
    RUN_TAG    = "_train2025_test2020_v7"
else:
    raise ValueError('DIRECTION must be "forward" or "reverse"')
N_BOOT     = 1000
SEED       = 42
REPEAT_SEEDS = [11, 22, 33, 44, 55]   # one full re-tuning per seed (progress saved after each)
N_TRIALS_REP = 40                      # Optuna trials per model in each repeat (the temporal run used 100)

ml_features = [
    'b4', 'bord', 'b0', 'hw1', 'first_birth', 'lbw', 'birth_weight',
    'bf_months', 'still_breastfeeding', 'v106', 'mother_bmi', 'mother_height', 'literacy',
    'anc_visits', 'anc_timing', 'delivery_facility', 'deworming',
    'v190', 'v025', 'v024', 'water_source_grp', 'water_time', 'toilet_grp', 'shared_toilet',
    'v481', 'distance_problem', 'v012', 'v136', 'v137', 'v404', 'iron_supp', 'm17', 'birth_size'
]
high_miss_vars = ['birth_weight', 'mother_bmi', 'mother_height', 'bf_months',
                  'anc_visits', 'anc_timing', 'delivery_facility', 'deworming']
categorical_vars = ['b4', 'b0', 'first_birth', 'v106', 'delivery_facility',
                    'v190', 'v025', 'v024', 'water_source_grp', 'toilet_grp',
                    'shared_toilet', 'v481', 'distance_problem', 'lbw',
                    'still_breastfeeding', 'deworming', 'v404', 'iron_supp',
                    'm17', 'birth_size']
continuous_vars = ['bord', 'hw1', 'birth_weight', 'bf_months', 'mother_bmi',
                   'mother_height', 'literacy', 'anc_visits', 'anc_timing',
                   'water_time', 'v012', 'v136', 'v137']

# ---- Optional sensitivity run (HARMONISED, see SETTINGS) ------------------------------------
# In 2019-20 the birth, antenatal, delivery and maternal-measurement items are recorded for nearly
# all children; in 2024-25 they are missing for 39-51 percent (children aged 36+ months, and a
# maternal anthropometry subsample). HARMONISED = True drops the 12 predictors whose
# missingness differs by more than 15 percentage points between the surveys.

# AGE_MAX (see SETTINGS) is applied to BOTH surveys: 24 keeps ages 0-23 months, 36 keeps 0-35 months.
HARMONISED_DROP = ['anc_timing', 'anc_visits', 'birth_size', 'birth_weight', 'delivery_facility',
                   'deworming', 'iron_supp', 'lbw', 'm17', 'mother_bmi', 'mother_height',
                   'still_breastfeeding']
if HARMONISED:
    ml_features      = [v for v in ml_features if v not in HARMONISED_DROP]
    high_miss_vars   = [v for v in high_miss_vars if v not in HARMONISED_DROP]
    categorical_vars = [v for v in categorical_vars if v not in HARMONISED_DROP]
    continuous_vars  = [v for v in continuous_vars if v not in HARMONISED_DROP]
    RUN_TAG = RUN_TAG + "_harmonised"
    print(f"HARMONISED run: {len(ml_features)} predictors kept ({len(HARMONISED_DROP)} dropped)")
if AGE_MAX is not None:
    RUN_TAG = RUN_TAG + f"_under{AGE_MAX}m"
    print(f"AGE restriction: children aged 0 to {AGE_MAX - 1} months only")


for _f in (TRAIN_FILE, TEST_FILE):
    if not os.path.exists(os.path.join(OUT_DIR, _f)):
        raise FileNotFoundError(f"Cannot find {_f} in {OUT_DIR}. Put the two .dta files in the same folder as this script.")
print("Using folder:", OUT_DIR)

def load(fname):
    # Read only the columns we use. Other columns (e.g. strata_id) can carry repeated value
    # labels, which makes pandas stop when it converts every column.
    d = pd.read_stata(os.path.join(OUT_DIR, fname), columns=list(dict.fromkeys(['cluster_id', 'stunted', 'hw1'] + ml_features)))
    if AGE_MAX is not None:
        d = d[pd.to_numeric(d['hw1'], errors='coerce') < AGE_MAX].copy()
        d = d[['cluster_id', 'stunted'] + ml_features]
    obs = set(d['stunted'].astype(str).str.strip().unique())
    if obs != {'Stunted', 'Not stunted'}:
        raise ValueError(f"{fname}: unexpected outcome labels {sorted(obs)}")
    y = (d['stunted'].astype(str).str.strip() == 'Stunted').astype(int).reset_index(drop=True)
    miss = [v for v in ml_features if v not in d.columns]
    if miss: raise KeyError(f"{fname}: missing predictors {miss}")
    empty = [v for v in ml_features if d[v].isna().all()]
    if empty: raise ValueError(f"{fname}: predictors entirely missing: {empty}")
    X = d[['cluster_id'] + ml_features].copy().reset_index(drop=True)
    for v in categorical_vars:                       # same text labels in both surveys
        s = X[v]
        X[v] = s.astype(object).where(s.notna(), None)
        X[v] = X[v].map(lambda z: None if (z is None or str(z).strip() in ('', 'nan', '.')) else str(z).strip().lower())
        if v == 'v024':   # 2019-20 and 2024-25 name the same five regions differently
            X[v] = X[v].replace({'city of kigali': 'kigali', 'southern province': 'south',
                                 'western province': 'west', 'northern province': 'north',
                                 'eastern province': 'east'})
    for v in continuous_vars:
        before = int(X[v].notna().sum())
        X[v] = pd.to_numeric(X[v], errors='coerce')
        if int(X[v].notna().sum()) != before:
            raise ValueError(f"{fname}: '{v}' lost values when converted to numbers; it may carry text labels.")
    for v in high_miss_vars:
        X[f'{v}_missing'] = X[v].isna().astype(int)
    return X, y

X_train_raw, y_train = load(TRAIN_FILE)
X_test_raw,  y_test  = load(TEST_FILE)
print(f"Train {TRAIN_FILE}: n={len(X_train_raw)}, stunted={int(y_train.sum())} ({y_train.mean():.3f})")
print(f"Test  {TEST_FILE}: n={len(X_test_raw)}, stunted={int(y_test.sum())} ({y_test.mean():.3f})")

# Fixed category lists taken from the TRAIN survey only, so train and test are encoded with
# exactly the same columns and the first category is always dropped. A test-survey category
# never seen in training is treated as missing and filled with the training mode.
CATS = {v: sorted(set(X_train_raw[v].dropna())) for v in categorical_vars}   # from the TRAIN survey only
for v in categorical_vars:
    only_tr = sorted(set(X_train_raw[v].dropna()) - set(X_test_raw[v].dropna()))
    only_te = sorted(set(X_test_raw[v].dropna()) - set(X_train_raw[v].dropna()))
    flag = ''
    if only_tr or only_te:
        flag = f"   <-- CHECK: only in train {only_tr}; only in test {only_te}"
    print(f"  {v}: {CATS[v]}{flag}")

# Stop if a categorical variable is coded differently in the two surveys: otherwise the test
# children are silently set to the training mode and the temporal result is biased.
for v in categorical_vars:
    share = float((~X_test_raw[v].dropna().isin(CATS[v])).mean()) if X_test_raw[v].notna().any() else 0.0
    if share > 0.02:
        raise ValueError(f"'{v}': {share:.0%} of test labels are not in the training labels. "
                         f"train={CATS[v]} test={sorted(set(X_test_raw[v].dropna()))}. "
                         f"Recode the labels so both surveys use the same text.")

def impute_and_encode(train_raw, test_raw):
    """Impute from the training data only, then one-hot with fixed categories."""
    tr, te = train_raw.copy(), test_raw.copy()
    for var in continuous_vars:
        med = tr[var].median()
        med = 0.0 if pd.isna(med) else med
        tr[var] = tr[var].fillna(med); te[var] = te[var].fillna(med)
    for var in categorical_vars:
        modes = tr[var].mode(dropna=True)
        mode = modes.iloc[0] if not modes.empty else CATS[var][0]
        unseen = te[var].notna() & ~te[var].isin(CATS[var])
        if unseen.any():
            print(f"  note: {int(unseen.sum())} test value(s) of {var} not seen in training; set to the training mode")
            te.loc[unseen, var] = None
        tr[var] = tr[var].fillna(mode); te[var] = te[var].fillna(mode)
    outs = []
    for d in (tr, te):
        d = d.drop(columns=['cluster_id']).copy()
        for var in categorical_vars:
            d[var] = pd.Categorical(d[var], categories=CATS[var])
        outs.append(pd.get_dummies(d, columns=categorical_vars, drop_first=True, dtype=float))
    X_tr, X_te = outs
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    if X_tr.isna().any().any() or X_te.isna().any().any():
        raise ValueError("NaN values remain after imputation/encoding.")
    return X_tr, X_te


# ---- MODELS, TUNING OBJECTIVE AND HELPERS ----
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, precision_recall_curve,
                              confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve

N_TRIALS = 100
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest',
               'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']

def make_objective(name, inner_splits):
    """Optuna objective: mean ROC AUC over every grouped inner fold."""
    def objective(trial):
        if name == 'Logistic Regression':
            params = {
                'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
            }
        elif name == 'Decision Tree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
            }
        elif name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'max_depth': trial.suggest_int('max_depth', 3, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        elif name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
        elif name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)
            }
        elif name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 12),
                'num_leaves': trial.suggest_int('num_leaves', 7, 127),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
            }
        elif name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'depth': trial.suggest_int('depth', 2, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True)
            }
        else:
            raise ValueError(f"Unknown model name: {name}")

        fold_aucs = []
        for X_tr, y_tr, X_val, y_val, spw in inner_splits:
            model = build_final_model(name, params)
            model = fit_with_weights(name, model, X_tr, y_tr, spw)
            probs = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, probs))
        return float(np.mean(fold_aucs))
    return objective

def build_final_model(name, params):
    if name == 'Logistic Regression':
        return LogisticRegression(**params, solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42)
    if name == 'Decision Tree':
        return DecisionTreeClassifier(**params, class_weight='balanced', random_state=42)
    if name == 'Random Forest':
        return RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1)
    if name == 'Gradient Boosting':
        return GradientBoostingClassifier(**params, random_state=42)
    if name == 'XGBoost':
        return XGBClassifier(**params, eval_metric='logloss', random_state=42, verbosity=0)
    if name == 'LightGBM':
        return LGBMClassifier(**params, class_weight='balanced', subsample_freq=1, random_state=42, verbose=-1)
    if name == 'CatBoost':
        return CatBoostClassifier(**params, auto_class_weights='Balanced', verbose=0, random_state=42)

def fit_with_weights(name, model, X_tr, y_tr, spw):
    """Fit a model, applying the model-appropriate imbalance handling."""
    if name == 'XGBoost':
        model.set_params(scale_pos_weight=spw)
        model.fit(X_tr, y_tr)
    elif name == 'Gradient Boosting':
        model.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))
    else:
        model.fit(X_tr, y_tr)
    return model

def select_threshold(probs_val, y_val):
    """Pick the F1-maximizing threshold from VALIDATION predictions only.
    Never call this with test-fold probabilities/labels — doing so leaks
    the test set into the threshold choice and inflates F1/precision/
    recall/accuracy (AUC is unaffected since it's threshold-free)."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx])


lab = lambda f: '2019-20' if '2020' in f else '2024-25'
TRAIN_LAB, TEST_LAB = lab(TRAIN_FILE), lab(TEST_FILE)
#======================================================================
# TEMPORAL SEED REPEATS: HOW MUCH DOES THE TEMPORAL AUC MOVE WITH THE TUNING AND MODEL SEEDS?
#======================================================================
# The temporal run tunes once (one inner partition, one Optuna seed, one model seed). Here the whole
# procedure is repeated with different seeds: new 4-fold grouped inner partition, new Optuna sampler seed,
# new random state for the final models, and a new F1 threshold from the inner folds. Every repeat trains on
# the TRAIN survey only and scores the TEST survey once; nothing is tuned on the test survey. The spread
# between repeats shows how much the temporal result depends on tuning and model randomness. It does NOT
# include sampling variability of the test survey (see the cluster-bootstrap intervals of the temporal run).
# Progress is saved after every repeat; if the run stops, start it again and it continues.
# No per-child values are saved.

import json, time
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.special import expit, logit

CKPT_DIR = os.path.join(OUT_DIR, "temporal_seed_repeats" + RUN_TAG)
os.makedirs(CKPT_DIR, exist_ok=True)
X_tr_all, X_te_all = impute_and_encode(X_train_raw, X_test_raw)
spw_all = float((y_train == 0).sum() / (y_train == 1).sum())
yt = y_test.to_numpy().astype(int)
EPS = 1e-6
def calib_cs(y, p):
    lp = logit(np.clip(p, EPS, 1 - EPS))
    a = 0.0
    for _ in range(100):
        mu = expit(a + lp); step = np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-12); a += step
        if abs(step) < 1e-10: break
    slope = float(LogisticRegression(C=1e10, solver='lbfgs', max_iter=2000).fit(lp.reshape(-1, 1), y).coef_[0][0])
    return float(a), slope

def with_seed(model, s):
    try:
        model.set_params(random_state=s)
    except Exception:
        pass
    return model

def run_seed(s):
    t0 = time.time()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=s)
    inner_splits = []
    for k, (itr, iva) in enumerate(inner.split(X_train_raw, y_train, groups=X_train_raw['cluster_id']), start=1):
        if set(X_train_raw['cluster_id'].iloc[itr]) & set(X_train_raw['cluster_id'].iloc[iva]):
            raise RuntimeError(f"Cluster leakage in inner fold {k}")
        y_itr, y_iva = y_train.iloc[itr].reset_index(drop=True), y_train.iloc[iva].reset_index(drop=True)
        X_itr, X_iva = impute_and_encode(X_train_raw.iloc[itr], X_train_raw.iloc[iva])
        inner_splits.append((X_itr, y_itr, X_iva, y_iva, float((y_itr == 0).sum() / (y_itr == 1).sum())))
    out = {'seed': s, 'models': {}}
    for name in MODEL_NAMES:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=4200 + s))
        study.optimize(make_objective(name, inner_splits), n_trials=N_TRIALS_REP, show_progress_bar=False)
        prm = study.best_params
        probs, ys = [], []
        for X_itr, y_itr, X_iva, y_iva, spw in inner_splits:
            m = fit_with_weights(name, with_seed(build_final_model(name, prm), s), X_itr, y_itr, spw)
            probs.extend(m.predict_proba(X_iva)[:, 1].tolist()); ys.extend(y_iva.tolist())
        thr = select_threshold(np.asarray(probs), np.asarray(ys))
        mdl = fit_with_weights(name, with_seed(build_final_model(name, prm), s), X_tr_all, y_train, spw_all)
        p = mdl.predict_proba(X_te_all)[:, 1]
        pred = (p >= thr).astype(int)
        a, sl = calib_cs(yt, p)
        out['models'][name] = {'auc': float(roc_auc_score(yt, p)), 'f1': float(f1_score(yt, pred, zero_division=0)),
                               'precision': float(precision_score(yt, pred, zero_division=0)),
                               'recall': float(recall_score(yt, pred, zero_division=0)), 'threshold': float(thr),
                               'calibration_intercept': a, 'calibration_slope': sl, 'mean_predicted_risk': float(p.mean()),
                               'inner_auc': float(study.best_value)}
        print(f"    seed {s}: {name} test AUC {out['models'][name]['auc']:.3f}", flush=True)
    out['minutes'] = (time.time() - t0) / 60
    return out

print(f"\nTemporal seed repeats: {len(REPEAT_SEEDS)} seeds x 7 models x {N_TRIALS_REP} tuning trials "
      f"(train {TRAIN_LAB} -> test {TEST_LAB}). Progress is saved in:\n  {CKPT_DIR}", flush=True)
for r_i, s in enumerate(REPEAT_SEEDS, start=1):
    path = os.path.join(CKPT_DIR, f"seed_{s}.json")
    if os.path.exists(path):
        print(f"Seed {s}: already finished, skipping.", flush=True)
        continue
    print(f"\n{'='*70}\nREPEAT {r_i}/{len(REPEAT_SEEDS)} (seed {s})\n{'='*70}", flush=True)
    res_s = run_seed(s)
    with open(path, 'w') as fh:
        json.dump(res_s, fh)
    print(f"Seed {s} saved ({res_s['minutes']:.1f} min).", flush=True)

files = sorted(f for f in os.listdir(CKPT_DIR) if f.startswith('seed_') and f.endswith('.json'))
reps = []
for f in files:
    with open(os.path.join(CKPT_DIR, f)) as fh:
        reps.append(json.load(fh))
R = len(reps)
print(f"\n{'='*100}\nTEMPORAL SEED REPEATS: {R} repeats, train {TRAIN_LAB} -> test {TEST_LAB}\n{'='*100}")
rows, per = [], []
for name in MODEL_NAMES:
    v = {k: np.array([r['models'][name][k] for r in reps]) for k in ('auc', 'f1', 'precision', 'recall', 'calibration_intercept', 'calibration_slope')}
    rows.append({'Model': name, 'Repeats': R, 'AUC_mean': v['auc'].mean(), 'AUC_SD_between_repeats': v['auc'].std(ddof=1) if R > 1 else np.nan,
                 'AUC_min': v['auc'].min(), 'AUC_max': v['auc'].max(), 'F1_mean': v['f1'].mean(), 'F1_SD': v['f1'].std(ddof=1) if R > 1 else np.nan,
                 'Recall_mean': v['recall'].mean(), 'Precision_mean': v['precision'].mean(),
                 'Calibration_intercept_mean': v['calibration_intercept'].mean(), 'Calibration_intercept_SD': v['calibration_intercept'].std(ddof=1) if R > 1 else np.nan,
                 'Calibration_slope_mean': v['calibration_slope'].mean(), 'Calibration_slope_SD': v['calibration_slope'].std(ddof=1) if R > 1 else np.nan})
    for r in reps:
        d = dict(r['models'][name]); d.update({'Model': name, 'Seed': r['seed']}); per.append(d)
summ = pd.DataFrame(rows); perdf = pd.DataFrame(per)
summ.to_csv(os.path.join(OUT_DIR, f"temporal_seed_repeats_summary{RUN_TAG}.csv"), index=False)
perdf.to_csv(os.path.join(OUT_DIR, f"temporal_seed_repeats_per_seed{RUN_TAG}.csv"), index=False)
with pd.option_context('display.width', 250, 'display.max_columns', 30):
    print(summ[['Model', 'AUC_mean', 'AUC_SD_between_repeats', 'AUC_min', 'AUC_max', 'F1_mean', 'Recall_mean', 'Precision_mean',
                'Calibration_intercept_mean', 'Calibration_slope_mean', 'Calibration_slope_SD']].round(3).to_string(index=False))
    print("\nTest-survey AUC of each repeat:")
    print(perdf.pivot(index='Model', columns='Seed', values='auc').reindex(MODEL_NAMES).round(3).to_string())
print(f"\n{'='*70}\nTEMPORAL SEED REPEATS COMPLETE\n{'='*70}")
print("Files written (aggregate tables only):")
print(f"   temporal_seed_repeats_summary{RUN_TAG}.csv\n   temporal_seed_repeats_per_seed{RUN_TAG}.csv")
