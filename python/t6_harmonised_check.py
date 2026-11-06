#======================================================================
# t6: QUICK CHECK OF THE HARMONISED RUN (CatBoost only)
#
# What it does: forces HARMONISED = True (drops the 12 predictors recorded very differently in the two surveys),
# prints how many predictors remain, tunes CatBoost on the TRAINING survey only (30 trials), and scores the TEST
# survey once. Use it to confirm that a harmonised run really dropped those predictors.
#
# How to run: put it in the same folder as t3_seed_repeats_train2020_test2025.py and the two .dta files.
# Output: t6_harmonised_check.txt (aggregate numbers only).
#======================================================================
import os, io, warnings
warnings.filterwarnings('ignore')
DIRECTION = "forward"   # "forward" or "reverse" (must match the t1 and t3 runs you are checking)
HERE = os.path.dirname(os.path.abspath(__file__))
F_T3 = os.path.join(HERE, 't3_seed_repeats_train2020_test2025.py')
src = open(F_T3, encoding='utf-8').read()
src = src.replace('DIRECTION  = "forward"', f'DIRECTION  = "{DIRECTION}"', 1)
assert 'HARMONISED = False' in src, "Could not find the HARMONISED switch in t3_seed_repeats_train2020_test2025.py"
src = src.replace('HARMONISED = False', 'HARMONISED = True', 1)
end = src.index("lab = lambda f:")                      # data + helper functions, before the repeat code
g = {'__file__': F_T3, '__name__': 'harm_part'}
exec(compile(src[:end], 't3_harmonised_part', 'exec'), g)

import numpy as np, optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

DROPPED = g['HARMONISED_DROP']
Xraw, y_tr, Xte_raw, y_te = g['X_train_raw'], g['y_train'].to_numpy().astype(int), g['X_test_raw'], g['y_test'].to_numpy().astype(int)
X_all_tr, X_all_te = g['impute_and_encode'](Xraw, Xte_raw)
still = [d for d in DROPPED if any(c == d or c.startswith(d + '_') for c in X_all_tr.columns)]

out = io.StringIO()
def say(s=''):
    print(s, flush=True); out.write(s + '\n')
say(f"HARMONISED = {g['HARMONISED']};  predictors kept = {len(g['ml_features'])};  encoded columns = {X_all_tr.shape[1]}")
say(f"Dropped predictors still present (should be empty): {still}")
if still or not g['HARMONISED']:
    raise SystemExit("Harmonised switch did not take effect.")

SEED, N_TR = 42, 30
splits = []
for itr, iva in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED).split(Xraw, y_tr, groups=Xraw['cluster_id']):
    a, b = g['impute_and_encode'](Xraw.iloc[itr], Xraw.iloc[iva])
    yi, yv = y_tr[itr], y_tr[iva]
    splits.append((a, yi, b, yv, float((yi == 0).sum() / (yi == 1).sum())))
import pandas as pd
splits = [(a, pd.Series(yi), b, pd.Series(yv), spw) for a, yi, b, yv, spw in splits]
st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=4200 + SEED))
st.optimize(g['make_objective']('CatBoost', splits), n_trials=N_TR, show_progress_bar=False)
m = g['fit_with_weights']('CatBoost', g['build_final_model']('CatBoost', st.best_params), X_all_tr, pd.Series(y_tr),
                          float((y_tr == 0).sum() / (y_tr == 1).sum()))
auc = roc_auc_score(y_te, m.predict_proba(X_all_te)[:, 1])
say(f"CatBoost harmonised: inner AUC {st.best_value:.4f}, TEST AUC {auc:.3f}  (earlier harmonised run: 0.687; full 33 predictors: 0.726)")
open(os.path.join(HERE, 't6_harmonised_check.txt'), 'w', encoding='utf-8').write(out.getvalue())
print('Saved t6_harmonised_check.txt (aggregate only; safe to share)')
