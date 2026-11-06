"""
Codebook and cross-survey check. Run BEFORE any analysis that uses both surveys.
Compares every variable in the two clean .dta files and flags: variables present in one file only,
categories that differ, different numeric ranges, and high missingness.
Writes codebook_comparison.csv (keep it on your computer; do not upload) and prints the flags.
Usage: python t0_codebook_check.py   (files in the same folder as this script)
"""
import os, pandas as pd, numpy as np

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
FILES = {"2019-20": "rwanda_stunting_clean_2020_v4.dta",
         "2024-25": "rwanda_stunting_clean_2025_b_v4.dta"}
MAX_LEVELS = 15   # up to this many distinct values a variable is treated as categorical

def read(fname):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put it in the same folder as this script.")
    # convert_categoricals=False keeps numeric codes; labels are read separately below
    codes = pd.read_stata(path, convert_categoricals=False)
    try:
        labels = pd.read_stata(path, convert_categoricals=True)
    except ValueError:                      # repeated value labels: fall back to codes
        labels = codes
    return codes, labels

data = {k: read(v) for k, v in FILES.items()}
(c1, l1), (c2, l2) = data["2019-20"], data["2024-25"]
rows, flags = [], []

for var in sorted(set(c1.columns) | set(c2.columns)):
    if var not in c1.columns or var not in c2.columns:
        flags.append(f"{var}: only in {'2019-20' if var in c1.columns else '2024-25'}")
        continue
    info = {"variable": var}
    cats = {}
    for s, c, l in (("2019-20", c1, l1), ("2024-25", c2, l2)):
        x, xl = c[var], l[var]
        info[f"n_{s}"] = len(x)
        info[f"missing_pct_{s}"] = round(100 * x.isna().mean(), 1)
        nun = x.nunique(dropna=True)
        is_cat = nun <= MAX_LEVELS or not pd.api.types.is_numeric_dtype(x)
        if is_cat:
            vc = xl.astype(str).str.strip().str.lower().value_counts(dropna=True)
            cats[s] = set(vc.index)
            info[f"values_{s}"] = "; ".join(f"{k}={v}" for k, v in vc.sort_index().items())
            code_vc = x.value_counts(dropna=True).sort_index()
            info[f"codes_{s}"] = ", ".join(str(k) for k in code_vc.index)
        else:
            info[f"values_{s}"] = f"min {x.min():.4g}, median {x.median():.4g}, max {x.max():.4g}"
            cats[s] = None
        info[f"type_{s}"] = "categorical" if is_cat else "numeric"
        if x.isna().mean() > 0.20:
            flags.append(f"{var}: {100*x.isna().mean():.0f}% missing in {s}")
    if info["type_2019-20"] != info["type_2024-25"]:
        flags.append(f"{var}: categorical in one survey, numeric in the other")
    elif cats["2019-20"] is not None and cats["2019-20"] != cats["2024-25"]:
        flags.append(f"{var}: LABELS DIFFER. only 2019-20 {sorted(cats['2019-20']-cats['2024-25'])}; "
                     f"only 2024-25 {sorted(cats['2024-25']-cats['2019-20'])}")
    elif cats["2019-20"] is None:
        a, b = c1[var].dropna(), c2[var].dropna()
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        span = (hi - lo) or 1
        if abs(a.median() - b.median()) > 0.5 * span or a.max() > 3 * b.max() + 1 or b.max() > 3 * a.max() + 1:
            flags.append(f"{var}: very different numeric range (check units or scaling)")
    rows.append(info)

out = pd.DataFrame(rows)
out_path = os.path.join(HERE, "codebook_comparison.csv")
out.to_csv(out_path, index=False)
print(f"{len(out)} variables compared. Codebook written to {out_path}")
print("\nFLAGS:" if flags else "\nNo flags.")
for f in flags:
    print("  -", f)
print("\nKeep codebook_comparison.csv on your computer; do not upload it.")
