#======================================================================
# t9: CODEBOOK CHECK OF THE 33 PREDICTORS IN BOTH SURVEY FILES (aggregate only)
# For each predictor and each survey it writes: stored label, type, % missing, minimum, maximum,
# number of distinct values, and the most frequent values with counts (value labels shown if the file has them).
# The point is to see, side by side, whether a variable is coded the same way (units, caps, categories) in both files.
# Put this file next to the two .dta files and run:  python t9_codebook_check.py
# Output: codebook_check.txt  (no per-child data; safe to share)
#======================================================================
import os, warnings, io
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
try: HERE = os.path.dirname(os.path.abspath(__file__))
except NameError: HERE = os.getcwd()
FILES = {'2019-20': "rwanda_stunting_clean_2020_v4.dta", '2024-25': "rwanda_stunting_clean_2025_b_v4.dta"}
VARS = ['b4','bord','b0','hw1','first_birth','lbw','birth_weight','bf_months','still_breastfeeding','v106','mother_bmi','mother_height','literacy','anc_visits','anc_timing','delivery_facility','deworming','v190','v025','v024','water_source_grp','water_time','toilet_grp','shared_toilet','v481','distance_problem','v012','v136','v137','v404','iron_supp','m17','birth_size']
out = io.StringIO()
def w(s=''): print(s); out.write(s + '\n')
info = {}
for lab, f in FILES.items():
    p = os.path.join(HERE, f)
    if not os.path.exists(p): raise FileNotFoundError(f"Cannot find {f} in {HERE}")
    rd = pd.io.stata.StataReader(p)
    labels = rd.variable_labels()
    raw = pd.read_stata(p, columns=VARS, convert_categoricals=False)     # numeric codes
    lab_df = pd.read_stata(p, columns=VARS, convert_categoricals=True)   # value labels where present
    info[lab] = (labels, raw, lab_df)
for v in VARS:
    w('=' * 100); w(v)
    for lab in FILES:
        labels, raw, lab_df = info[lab]
        x = raw[v]; xl = lab_df[v]
        w(f"  [{lab}] label: {labels.get(v, '')!r}; stored as {x.dtype}; missing {100*x.isna().mean():.1f}%")
        num = pd.to_numeric(x, errors='coerce')
        if num.notna().any():
            w(f"      min {num.min():g}  max {num.max():g}  distinct {num.nunique()}")
        vc = xl.value_counts(dropna=True).head(12)
        if x.dtype == object or num.nunique() <= 12:
            w("      values: " + "; ".join(f"{k} = {n}" for k, n in vc.items()))
        else:
            top = num.value_counts().head(8)
            w("      most frequent: " + "; ".join(f"{k:g} = {n}" for k, n in top.items()) + f";  highest values: " + ", ".join(f"{k:g} (n={n})" for k, n in num.value_counts().sort_index(ascending=False).head(5).items()))
open(os.path.join(HERE, 'codebook_check.txt'), 'w', encoding='utf-8').write(out.getvalue())
print("\nWrote codebook_check.txt (aggregate only).")
