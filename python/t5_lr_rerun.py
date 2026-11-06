#======================================================================
# t5: LOGISTIC REGRESSION RE-RUN WITH THE TUNED C PRINTED
#
# What it does: re-tunes logistic regression only (100 Optuna trials, inner folds on the TRAINING survey) for the
# main seed (42) and the five repeat seeds, and prints the chosen penalty and C. It also reports a rule fixed in
# advance from inner CV only: the most regularised setting within 1 standard error of the best inner AUC.
# The test survey is scored once per fit and is never used to choose anything.
#
# How to run: put it in the same folder as t3_seed_repeats_train2020_test2025.py and the two .dta files.
# Output: t5_lr_rerun.txt (aggregate numbers only).
#======================================================================
import os, io, warnings
warnings.filterwarnings('ignore')
DIRECTION = "forward"   # "forward" or "reverse" (must match the t1 and t3 runs you are checking)
HERE = os.path.dirname(os.path.abspath(__file__))
F_T3 = os.path.join(HERE, 't3_seed_repeats_train2020_test2025.py')
src = open(F_T3, encoding='utf-8').read()
src = src.replace('DIRECTION  = "forward"', f'DIRECTION  = "{DIRECTION}"', 1)
cut = src.index('# ---- MODELS, TUNING OBJECTIVE AND HELPERS')
g = {'__file__': F_T3, '__name__': 'data_part'}
exec(compile(src[:cut], 't3_data_part', 'exec'), g)

import numpy as np, pandas as pd, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

impute_and_encode = g['impute_and_encode']
Xraw, yS, Xte_raw = g['X_train_raw'], g['y_train'], g['X_test_raw']
y_tr, y_te = yS.to_numpy().astype(int), g['y_test'].to_numpy().astype(int)
X_all_tr, X_all_te = impute_and_encode(Xraw, Xte_raw)
N_TRIALS = 100
SEEDS = [42, 11, 22, 33, 44, 55]     # 42 = the seed used in the main script; the rest = the repeat seeds

def lr(C, pen):
    return LogisticRegression(C=C, penalty=pen, solver='liblinear', class_weight='balanced',
                              max_iter=1000, random_state=42)

out = io.StringIO()
def say(s=''):
    print(s, flush=True); out.write(s + '\n')

rows = []
for s in SEEDS:
    splits = []
    for itr, iva in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=s).split(Xraw, y_tr, groups=Xraw['cluster_id']):
        a, b = impute_and_encode(Xraw.iloc[itr], Xraw.iloc[iva])
        splits.append((a, y_tr[itr], b, y_tr[iva]))
    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        pen = trial.suggest_categorical('penalty', ['l1', 'l2'])
        aucs = [roc_auc_score(yv, lr(C, pen).fit(a, yt).predict_proba(b)[:, 1]) for a, yt, b, yv in splits]
        trial.set_user_attr('se', float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))))
        return float(np.mean(aucs))
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=4200 + s))
    st.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = st.best_trial
    p = lr(best.params['C'], best.params['penalty']).fit(X_all_tr, y_tr).predict_proba(X_all_te)[:, 1]
    # one-SE rule (inner CV only): most regularised (smallest C) trial with inner AUC >= best - SE(best)
    ok = [t for t in st.trials if t.value >= best.value - best.user_attrs['se']]
    t1 = min(ok, key=lambda t: t.params['C'])
    p1 = lr(t1.params['C'], t1.params['penalty']).fit(X_all_tr, y_tr).predict_proba(X_all_te)[:, 1]
    rows.append(dict(seed=s, tuned_penalty=best.params['penalty'], tuned_C=round(best.params['C'], 3),
                     inner_auc=round(best.value, 4), test_auc=round(roc_auc_score(y_te, p), 4),
                     mean_test_risk=round(float(p.mean()), 3),
                     oneSE_penalty=t1.params['penalty'], oneSE_C=round(t1.params['C'], 3),
                     oneSE_inner_auc=round(t1.value, 4), oneSE_test_auc=round(roc_auc_score(y_te, p1), 4)))
    say(str(rows[-1]))

df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
say('\n=== LR re-run, forward 2019-20 -> 2024-25, 100 trials per seed ===')
say(df.to_string(index=False))
say(f"\nTuned (best inner AUC): test AUC mean {df.test_auc.mean():.3f}, range {df.test_auc.min():.3f} to {df.test_auc.max():.3f}")
say(f"One-SE rule           : test AUC mean {df.oneSE_test_auc.mean():.3f}, range {df.oneSE_test_auc.min():.3f} to {df.oneSE_test_auc.max():.3f}")
open(os.path.join(HERE, 't5_lr_rerun.txt'), 'w', encoding='utf-8').write(out.getvalue())
print('\nSaved t5_lr_rerun.txt (aggregate only; safe to share)')
