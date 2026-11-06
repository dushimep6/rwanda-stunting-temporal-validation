# PRESET: train on 2024-25, test on 2019-20 (reverse).
#======================================================================
# t7: EXTRA ANALYSES RAISED IN AN INDEPENDENT REVIEW OF THE DRAFT
#
# Answers five review concerns, in one run per direction. Nothing is tuned on the test survey except the
# "within-survey" comparison in Part D, which by design trains inside the test survey (as the earlier
# within-survey cross-validation did). All other models train on the TRAINING survey only.
#
#   Part A  Offset inside the training survey (concern 1b): calibration of the weighted models on the
#           pooled inner out-of-fold predictions of the training survey, next to the test-survey offset.
#   Part B  Models WITHOUT class weights (concern 1c): CatBoost, XGBoost, logistic regression.
#   Part C  Paired cluster-bootstrap AUC differences between the models (concern 4).
#   Part D  Transported AUC minus within-survey AUC, with a paired interval (concern 3).
#   Part E  How much labelled data recalibration needs (concern 2).
#   Part F  SHAP comparison of XGBoost trained separately on each survey (concern 7).
#
# How to run: put this file in the same folder as t3_seed_repeats_train2020_test2025.py and the two .dta files.
#   DIRECTION is already set to match the file name, so no editing is needed. Run each file once.
#   Test it quickly first:  set the environment variable T7_QUICK=1 (3 tuning trials, 60 bootstraps).
# Time: several hours at full size (Parts A, B and D tune many models). Progress is cached in t7_cache/
#   (local files, never upload them: they hold per-child predictions). If it stops, start it again.
# Output: t7_review_analyses_<direction>.txt (aggregate numbers only, safe to share).
#======================================================================
import os, io, sys, pickle, warnings, time
warnings.filterwarnings('ignore')
DIRECTION = "reverse"   # "forward" or "reverse"
QUICK = bool(os.environ.get('T7_QUICK'))
N_MAIN   = 3 if QUICK else 100    # tuning trials, weighted models (Part A; same as the main run)
N_UNW    = 3 if QUICK else 40     # tuning trials, unweighted models (Part B)
N_WITHIN = 3 if QUICK else 30     # tuning trials per outer fold, within-survey models (Part D)
N_SHAP   = 3 if QUICK else 40     # tuning trials, XGBoost trained on the test survey (Part F)
N_BOOT   = 60 if QUICK else 1000  # cluster-bootstrap resamples (Parts C and D)
N_RECAL_REPS = 20 if QUICK else 300
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
F_T3 = os.path.join(HERE, 't3_seed_repeats_train2020_test2025.py')
src = open(F_T3, encoding='utf-8').read()
assert 'DIRECTION  = "forward"' in src, "Could not find the DIRECTION line in t3_seed_repeats_train2020_test2025.py"
src = src.replace('DIRECTION  = "forward"', f'DIRECTION  = "{DIRECTION}"', 1)
end = src.index("lab = lambda f:")                       # data + model helpers (no repeat code)
g = {'__file__': F_T3, '__name__': 't7_part'}
exec(compile(src[:end], 't3_data_part', 'exec'), g)

import numpy as np, pandas as pd, optuna
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
optuna.logging.set_verbosity(optuna.logging.WARNING)

Xraw, Xte_raw = g['X_train_raw'], g['X_test_raw']
y_tr = g['y_train'].reset_index(drop=True); y_te = g['y_test'].reset_index(drop=True)
Xraw = Xraw.reset_index(drop=True); Xte_raw = Xte_raw.reset_index(drop=True)
yte = y_te.to_numpy().astype(int); ytr = y_tr.to_numpy().astype(int)
cl_te = Xte_raw['cluster_id'].to_numpy()
impute_and_encode, make_objective = g['impute_and_encode'], g['make_objective']
build_weighted, fit_weighted = g['build_final_model'], g['fit_with_weights']
select_threshold, MODEL_NAMES = g['select_threshold'], g['MODEL_NAMES']
TR_LAB, TE_LAB = g['TRAIN_FILE'], g['TEST_FILE']
lab = lambda f: '2019-20' if '2020' in f else '2024-25'
TR_LAB, TE_LAB = lab(TR_LAB), lab(TE_LAB)
CACHE = os.path.join(HERE, 't7_cache' + ('_quick' if QUICK else '') + '_' + DIRECTION); os.makedirs(CACHE, exist_ok=True)

out = io.StringIO()
def say(s=''):
    print(s, flush=True); out.write(s + '\n')

# ------------------------------ helpers ------------------------------
EPS = 1e-6
def offset_only(y, p):
    lp = logit(np.clip(p, EPS, 1 - EPS)); a = 0.0
    for _ in range(100):
        mu = expit(a + lp); step = np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-12); a += step
        if abs(step) < 1e-10: break
    return float(a)
def slope_of(y, p):
    lp = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
    return float(LogisticRegression(C=1e10, solver='lbfgs', max_iter=2000).fit(lp, y).coef_[0][0])
def ece10(y, p):
    o = np.argsort(p); bins = np.array_split(o, 10)
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins) / len(p))
def calib(y, p):
    return dict(mean_pred=float(p.mean()), obs=float(y.mean()), citl=offset_only(y, p), slope=slope_of(y, p),
                brier=float(np.mean((p - y) ** 2)), ece=ece10(y, p))
def pct(a): return np.percentile(a, [2.5, 97.5])

# unweighted versions of the model builders (class weights switched off)
def build_unweighted(name, params):
    m = build_weighted(name, params)
    if name in ('Logistic Regression', 'Decision Tree', 'Random Forest', 'LightGBM'): m.set_params(class_weight=None)
    if name == 'CatBoost': m.set_params(auto_class_weights='None')
    return m
def fit_unweighted(name, model, X, y, spw):
    if name == 'XGBoost': model.set_params(scale_pos_weight=1.0)
    model.fit(X, y); return model

def tune(name, raw, y, n_trials, weighted=True, seed=SEED):
    """Tune on `raw` only (4 cluster-grouped inner folds). Returns params, pooled out-of-fold probabilities, F1 threshold."""
    builder, fitter = (build_weighted, fit_weighted) if weighted else (build_unweighted, fit_unweighted)
    g['build_final_model'], g['fit_with_weights'] = builder, fitter       # make_objective looks these up in g
    try:
        splits, idxs = [], []
        for itr, iva in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(raw, y, groups=raw['cluster_id']):
            a, b = impute_and_encode(raw.iloc[itr], raw.iloc[iva])
            yi, yv = y.iloc[itr].reset_index(drop=True), y.iloc[iva].reset_index(drop=True)
            splits.append((a, yi, b, yv, float((yi == 0).sum() / max((yi == 1).sum(), 1)))); idxs.append(iva)
        st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=4200 + seed))
        st.optimize(make_objective(name, splits), n_trials=n_trials, show_progress_bar=False)
        prm = st.best_params
        oof = np.zeros(len(raw)); 
        for (a, yi, b, yv, spw), iva in zip(splits, idxs):
            m = fitter(name, builder(name, prm), a, yi, spw); oof[iva] = m.predict_proba(b)[:, 1]
        thr = select_threshold(oof, y.to_numpy().astype(int))
    finally:
        g['build_final_model'], g['fit_with_weights'] = build_weighted, fit_weighted
    return prm, oof, thr, float(st.best_value)

def fit_final(name, prm, Xa, ya, weighted=True):
    spw = float((ya == 0).sum() / max((ya == 1).sum(), 1))
    if weighted: return fit_weighted(name, build_weighted(name, prm), Xa, ya, spw)
    return fit_unweighted(name, build_unweighted(name, prm), Xa, ya, spw)

def cached(key, fn):
    path = os.path.join(CACHE, key + '.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    r = fn(); pickle.dump(r, open(path, 'wb')); return r

X_tr_all, X_te_all = impute_and_encode(Xraw, Xte_raw)
say(f"t7 review analyses | {DIRECTION}: train {TR_LAB} (n={len(Xraw)}, stunted {ytr.mean():.3f}) -> test {TE_LAB} (n={len(Xte_raw)}, stunted {yte.mean():.3f})")
say(f"Settings: trials main/unweighted/within/shap = {N_MAIN}/{N_UNW}/{N_WITHIN}/{N_SHAP}; bootstraps {N_BOOT}; recal repeats {N_RECAL_REPS}" + ("  [QUICK TEST RUN]" if QUICK else ""))

# ============ PART A: weighted models, test predictions, training-survey offset ============
def do_weighted(name):
    prm, oof, thr, inner = tune(name, Xraw, y_tr, N_MAIN, True)
    m = fit_final(name, prm, X_tr_all, y_tr, True)
    return dict(prm=prm, oof=oof, thr=thr, inner=inner, p=m.predict_proba(X_te_all)[:, 1])
W = {}
for name in MODEL_NAMES:
    t0 = time.time(); W[name] = cached('weighted_' + name.replace(' ', '_'), lambda n=name: do_weighted(n))
    print(f"   weighted {name}: done ({(time.time()-t0)/60:.1f} min if not cached)", flush=True)

say("\n" + "=" * 100 + "\nPART A. CALIBRATION OFFSET INSIDE THE TRAINING SURVEY (pooled inner out-of-fold) VERSUS THE TEST SURVEY\n" + "=" * 100)
say(f"Class weighting alone is expected to give a calibration-in-the-large (CITL) near logit(prevalence): training {logit(ytr.mean()):.2f}, test {logit(yte.mean()):.2f}.")
rows = []
for name in MODEL_NAMES:
    a = calib(ytr, W[name]['oof']); b = calib(yte, W[name]['p'])
    rows.append(dict(Model=name, Train_mean_pred=a['mean_pred'], Train_obs=a['obs'], Train_CITL=a['citl'], Train_slope=a['slope'],
                     Test_mean_pred=b['mean_pred'], Test_obs=b['obs'], Test_CITL=b['citl'], Test_slope=b['slope'], Shift_in_CITL=b['citl'] - a['citl']))
say(pd.DataFrame(rows).round(3).to_string(index=False))
say("Reading: if Test_CITL is close to Train_CITL (Shift_in_CITL near 0 after allowing for the prevalence difference), the overestimate comes mostly from class weighting, not from a change between surveys.")
say(f"Expected CITL change from the prevalence difference alone: {logit(yte.mean()) - logit(ytr.mean()):+.2f}")

# ============ PART B: unweighted models ============
say("\n" + "=" * 100 + "\nPART B. MODELS TRAINED WITHOUT CLASS WEIGHTS (trained on the training survey only)\n" + "=" * 100)
rows = []; UN = {}
for name in ['CatBoost', 'XGBoost', 'Logistic Regression']:
    def do_unw(n=name):
        prm, oof, thr, inner = tune(n, Xraw, y_tr, N_UNW, False)
        m = fit_final(n, prm, X_tr_all, y_tr, False)
        return dict(prm=prm, oof=oof, p=m.predict_proba(X_te_all)[:, 1], inner=inner)
    UN[name] = cached('unweighted_' + name.replace(' ', '_'), do_unw)
    for tag, d in (('Weighted', W[name]), ('Unweighted', UN[name])):
        c = calib(yte, d['p']); ct = calib(ytr, d['oof'])
        rows.append(dict(Model=name, Training=tag, Test_AUC=roc_auc_score(yte, d['p']), Test_mean_pred=c['mean_pred'], Test_obs=c['obs'],
                         Test_CITL=c['citl'], Test_slope=c['slope'], Test_Brier=c['brier'], Test_ECE=c['ece'], Train_CITL=ct['citl']))
say(pd.DataFrame(rows).round(3).to_string(index=False))
say("Reading: unweighted models should show Test_CITL near 0 if class weighting caused the overestimate; any remaining offset is a real change between surveys (or tuning noise). Brier and ECE compare calibration directly.")

# ============ shared cluster bootstrap indices ============
rng = np.random.default_rng(SEED)
ucl = np.unique(cl_te); pos = {c: np.where(cl_te == c)[0] for c in ucl}
BOOT = [np.concatenate([pos[c] for c in rng.choice(ucl, len(ucl), replace=True)]) for _ in range(N_BOOT)]
def paired_auc_diff(p1, p2):
    d = []
    for ix in BOOT:
        yy = yte[ix]
        if yy.min() == yy.max(): continue
        d.append(roc_auc_score(yy, p1[ix]) - roc_auc_score(yy, p2[ix]))
    d = np.array(d); lo, hi = pct(d); return float(roc_auc_score(yte, p1) - roc_auc_score(yte, p2)), float(lo), float(hi)

# ============ PART C: paired AUC differences ============
say("\n" + "=" * 100 + f"\nPART C. PAIRED CLUSTER-BOOTSTRAP AUC DIFFERENCES IN THE TEST SURVEY ({N_BOOT} resamples)\n" + "=" * 100)
say("Difference = AUC(CatBoost) minus AUC(other model). An interval that contains 0 means the two models cannot be told apart.")
rows = []
for name in MODEL_NAMES:
    if name == 'CatBoost': continue
    d, lo, hi = paired_auc_diff(W['CatBoost']['p'], W[name]['p'])
    rows.append(dict(Other_model=name, AUC_CatBoost=roc_auc_score(yte, W['CatBoost']['p']), AUC_other=roc_auc_score(yte, W[name]['p']), Difference=d, CI_low=lo, CI_high=hi,
                     Interval_contains_0='yes' if lo <= 0 <= hi else 'no'))
say(pd.DataFrame(rows).round(3).to_string(index=False))

# ============ PART D: transported versus within-survey AUC ============
say("\n" + "=" * 100 + f"\nPART D. TRANSPORTED MODEL VERSUS MODEL TRAINED WITHIN THE TEST SURVEY ({TE_LAB})\n" + "=" * 100)
say(f"Within-survey = 5-fold cluster-grouped cross-validation inside the test survey, tuned on each outer-training part ({N_WITHIN} trials, 4 inner folds).")
say("Difference = AUC(transported) minus AUC(within-survey), on the same children, paired cluster bootstrap. Positive = transported model did better.")
def do_within(name):
    oof = np.zeros(len(Xte_raw))
    for k, (itr, iva) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xte_raw, y_te, groups=Xte_raw['cluster_id']), 1):
        a_raw, b_raw = Xte_raw.iloc[itr].reset_index(drop=True), Xte_raw.iloc[iva].reset_index(drop=True)
        ya = y_te.iloc[itr].reset_index(drop=True)
        prm, _, _, _ = tune(name, a_raw, ya, N_WITHIN, True)
        A, B = impute_and_encode(a_raw, b_raw)
        oof[iva] = fit_final(name, prm, A, ya, True).predict_proba(B)[:, 1]
        print(f"   within-survey {name}: outer fold {k}/5 done", flush=True)
    return oof
rows = []
for name in ['CatBoost', 'XGBoost', 'Logistic Regression']:
    pw = cached('within_' + name.replace(' ', '_'), lambda n=name: do_within(n))
    d, lo, hi = paired_auc_diff(W[name]['p'], pw)
    rows.append(dict(Model=name, AUC_transported=roc_auc_score(yte, W[name]['p']), AUC_within_survey=roc_auc_score(yte, pw), Difference=d, CI_low=lo, CI_high=hi,
                     Interval_contains_0='yes' if lo <= 0 <= hi else 'no'))
say(pd.DataFrame(rows).round(3).to_string(index=False))

# ============ PART E: how much labelled data does recalibration need? ============
say("\n" + "=" * 100 + f"\nPART E. RECALIBRATION WITH A SMALL LABELLED SAMPLE FROM THE NEW SURVEY ({N_RECAL_REPS} random splits of clusters)\n" + "=" * 100)
say("For each fraction, that share of test-survey clusters is used to fit the recalibration; the remaining clusters are the evaluation set. Values are means over splits (2.5th-97.5th percentile for ECE).")
def recal_eval(p, frac, rng_):
    cs = rng_.permutation(ucl); k = max(int(round(frac * len(ucl))), 1)
    fit_ix = np.concatenate([pos[c] for c in cs[:k]]); ev_ix = np.concatenate([pos[c] for c in cs[k:]])
    yf, pf, ye, pe = yte[fit_ix], p[fit_ix], yte[ev_ix], p[ev_ix]
    if yf.min() == yf.max() or ye.min() == ye.max(): return None
    a = offset_only(yf, pf)
    lpf, lpe = logit(np.clip(pf, EPS, 1 - EPS)), logit(np.clip(pe, EPS, 1 - EPS))
    pl = LogisticRegression(C=1e10, solver='lbfgs', max_iter=2000).fit(lpf.reshape(-1, 1), yf)
    res = {'Raw': pe, 'Intercept only': expit(lpe + a), 'Intercept and slope': pl.predict_proba(lpe.reshape(-1, 1))[:, 1]}
    return {k_: (float(np.mean((v - ye) ** 2)), ece10(ye, v), offset_only(ye, v)) for k_, v in res.items()}
rows = []
for name in ['CatBoost', 'XGBoost', 'Logistic Regression']:
    for frac in (0.05, 0.10, 0.25, 0.50):
        r_ = np.random.default_rng(SEED); acc = {}
        for _ in range(N_RECAL_REPS):
            o = recal_eval(W[name]['p'], frac, r_)
            if o is None: continue
            for k_, v in o.items(): acc.setdefault(k_, []).append(v)
        for k_, v in acc.items():
            v = np.array(v); lo, hi = pct(v[:, 1])
            rows.append(dict(Model=name, Fraction_clusters=frac, Version=k_, Brier=v[:, 0].mean(), ECE=v[:, 1].mean(), ECE_lo=lo, ECE_hi=hi, CITL=v[:, 2].mean()))
say(pd.DataFrame(rows).round(3).to_string(index=False))
say("Reading: find the smallest fraction at which ECE and Brier for the recalibrated versions are close to the full-sample values in Table 3 of the manuscript.")

# ============ PART F: SHAP of XGBoost trained separately on each survey ============
say("\n" + "=" * 100 + "\nPART F. SHAP OF XGBOOST TRAINED SEPARATELY ON EACH SURVEY, EXPLAINED IN THE SAME SURVEY\n" + "=" * 100)
import shap
def do_shap():
    mA = fit_final('XGBoost', W['XGBoost']['prm'], X_tr_all, y_tr, True)           # trained on the training survey
    prmB, _, _, _ = tune('XGBoost', Xte_raw, y_te, N_SHAP, True)                    # tuned and trained on the test survey
    XB_tr, XB_on_tr = impute_and_encode(Xte_raw, Xraw)                              # test-survey rules
    mB = fit_final('XGBoost', prmB, XB_tr, y_te, True)
    sub = np.random.default_rng(SEED).choice(len(X_te_all), size=min(2000, len(X_te_all)), replace=False)
    sA = np.abs(shap.TreeExplainer(mA).shap_values(X_te_all.iloc[sub])).mean(0)
    sB = np.abs(shap.TreeExplainer(mB).shap_values(XB_tr.iloc[sub])).mean(0)
    return dict(cols=list(X_te_all.columns), sA=sA, sB=sB)
S = cached('shap_separate', do_shap)
rho = spearmanr(S['sA'], S['sB'])[0]
rA, rB = pd.Series(S['sA'], S['cols']).rank(ascending=False), pd.Series(S['sB'], S['cols']).rank(ascending=False)
top = lambda s, k: set(pd.Series(s, S['cols']).sort_values(ascending=False).index[:k])
say(f"Model A = trained on {TR_LAB}; model B = trained on {TE_LAB}; both explained on {TE_LAB} children (up to 2,000, each with its own preprocessing rules).")
say(f"Spearman correlation of mean |SHAP| over {len(S['cols'])} encoded features: {rho:.3f}")
say(f"Overlap of the top 5 features: {len(top(S['sA'],5)&top(S['sB'],5))}/5; top 10: {len(top(S['sA'],10)&top(S['sB'],10))}/10")
tab = pd.DataFrame({'Feature': S['cols'], 'SHAP_A': S['sA'], 'Rank_A': rA.values, 'SHAP_B': S['sB'], 'Rank_B': rB.values}).sort_values('Rank_A').head(15)
say(tab.round(3).to_string(index=False))
say("Reading: a high correlation means models trained separately on each survey rely on the same features; this goes beyond the earlier check that explained ONE model in both surveys.")

say("\nDONE. Aggregate numbers only; safe to share.")
open(os.path.join(HERE, f't7_review_analyses_{DIRECTION}' + ('_QUICK' if QUICK else '') + '.txt'), 'w', encoding='utf-8').write(out.getvalue())
print('Saved results text file.')
