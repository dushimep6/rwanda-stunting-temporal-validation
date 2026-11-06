#======================================================================
# t4: LOGISTIC REGRESSION GRID (forward 2019-20 -> 2024-25)
#
# Why: logistic regression gave a lower temporal AUC in the main run than in the seed repeats. This script
# shows how the TEST AUC changes with the penalty (l1, l2) and C, next to the inner-CV AUC that the tuning sees.
# It is a diagnostic. Do NOT pick C from this table: that would be tuning on the test survey.
#
# How to run: put it in the same folder as t3_seed_repeats_train2020_test2025.py and the two .dta files.
# Output: t4_lr_grid.txt (aggregate numbers only).
#======================================================================
import os, sys, io, warnings
warnings.filterwarnings('ignore')
DIRECTION = "forward"   # "forward" or "reverse" (must match the t1 and t3 runs you are checking)
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 't3_seed_repeats_train2020_test2025.py'), encoding='utf-8').read()
cut = src.index('# ---- MODELS, TUNING OBJECTIVE AND HELPERS')
g = {'__file__': os.path.join(HERE, 't3_seed_repeats_train2020_test2025.py'), '__name__': 'data_part'}
exec(compile(src[:cut], 't3_data_part', 'exec'), g)

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

X_tr, X_te = g['impute_and_encode'](g['X_train_raw'], g['X_test_raw'])
y_tr, y_te = g['y_train'].values, g['y_test'].values
groups = g['X_train_raw']['cluster_id'].values
cont = [c for c in g['continuous_vars'] if c in X_tr.columns]

def scaled(cols_to_scale):
    a, b = X_tr.copy(), X_te.copy()
    sc = StandardScaler().fit(a[cols_to_scale])
    a[cols_to_scale] = sc.transform(a[cols_to_scale]); b[cols_to_scale] = sc.transform(b[cols_to_scale])
    return a, b

def run(Xa, Xb, label, seeds=(11, 22, 33, 44, 55)):
    rows = []
    for pen in ['l1', 'l2']:
        for C in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10]:
            inner = []
            for s in seeds:
                folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=s).split(Xa, y_tr, groups)
                aucs = []
                for i, j in folds:
                    m = LogisticRegression(C=C, penalty=pen, solver='liblinear', class_weight='balanced',
                                           max_iter=1000, random_state=42).fit(Xa.iloc[i], y_tr[i])
                    aucs.append(roc_auc_score(y_tr[j], m.predict_proba(Xa.iloc[j])[:, 1]))
                inner.append(np.mean(aucs))
            m = LogisticRegression(C=C, penalty=pen, solver='liblinear', class_weight='balanced',
                                   max_iter=1000, random_state=42).fit(Xa, y_tr)
            p = m.predict_proba(Xb)[:, 1]
            rows.append(dict(setup=label, penalty=pen, C=C, inner_auc=round(np.mean(inner), 4),
                             inner_sd_over_seeds=round(np.std(inner), 4),
                             test_auc=round(roc_auc_score(y_te, p), 4),
                             nonzero_coefs=int((m.coef_ != 0).sum()), n_coefs=m.coef_.size,
                             mean_test_risk=round(p.mean(), 3)))
    return pd.DataFrame(rows)

out = io.StringIO()
def say(s=''):
    print(s); out.write(s + '\n')

say(f"Train n={len(y_tr)}, test n={len(y_te)}, predictors after encoding={X_tr.shape[1]}")
say("Continuous predictors (unscaled ranges in training):")
for c in cont:
    say(f"  {c}: min {X_tr[c].min():.2f}, max {X_tr[c].max():.2f}, sd {X_tr[c].std():.2f}")

pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
A = run(X_tr, X_te, 'AS_IN_PAPER (continuous unscaled)')
say('\n=== A. As in the paper: continuous predictors NOT standardised ===')
say(A.drop(columns='setup').to_string(index=False))
say(f"\nBest inner AUC row: {A.loc[A.inner_auc.idxmax()].to_dict()}")
say(f"Range of test AUC over C>=0.01 : {A[A.C>=0.01].test_auc.min()} to {A[A.C>=0.01].test_auc.max()}")

a, b = scaled(cont)
B = run(a, b, 'STANDARDISED continuous')
say('\n=== B. Same grid with continuous predictors standardised (train mean/sd) ===')
say(B.drop(columns='setup').to_string(index=False))
say(f"\nBest inner AUC row: {B.loc[B.inner_auc.idxmax()].to_dict()}")

open(os.path.join(HERE, 't4_lr_grid.txt'), 'w', encoding='utf-8').write(out.getvalue())
print('\nSaved t4_lr_grid.txt  (aggregate only; safe to share)')
