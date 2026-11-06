#======================================================================
# t8: SIDE-BY-SIDE PREDICTOR TABLE FOR THE TWO SURVEYS (TRIPOD+AI items 20b and 20c)
#
# For every one of the 33 candidate predictors it writes, for 2019-20 and 2024-25:
#   - percentage missing,
#   - for numbers: mean (SD); for categories: percentage in each category (of children with a value).
# Output: predictor_table_by_survey.csv  (aggregate only; nothing per child; safe to share)
# Put this file in the same folder as the two .dta files and run it:  python t8_predictor_table.py
#======================================================================
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
try: HERE = os.path.dirname(os.path.abspath(__file__))
except NameError: HERE = os.getcwd()
F20, F25 = "rwanda_stunting_clean_2020_v4.dta", "rwanda_stunting_clean_2025_b_v4.dta"
ml_features = ['b4','bord','b0','hw1','first_birth','lbw','birth_weight','bf_months','still_breastfeeding','v106','mother_bmi','mother_height','literacy','anc_visits','anc_timing','delivery_facility','deworming','v190','v025','v024','water_source_grp','water_time','toilet_grp','shared_toilet','v481','distance_problem','v012','v136','v137','v404','iron_supp','m17','birth_size']
categorical = ['b4','b0','first_birth','v106','delivery_facility','v190','v025','v024','water_source_grp','toilet_grp','shared_toilet','v481','distance_problem','lbw','still_breastfeeding','deworming','v404','iron_supp','m17','birth_size']
dropped = ['anc_timing','anc_visits','birth_size','birth_weight','delivery_facility','deworming','iron_supp','lbw','m17','mother_bmi','mother_height','still_breastfeeding']
# Short descriptions: my reading of the DHS variable names. Please check them against your codebook.
desc = {'b4':'Child sex','bord':'Birth order','b0':'Multiple birth','hw1':"Child's age (months)",'first_birth':'First birth','lbw':'Low birth weight','birth_weight':'Birth weight','bf_months':'Months breastfed','still_breastfeeding':'Child still being breastfed','v106':"Mother's education",'mother_bmi':"Mother's BMI",'mother_height':"Mother's height",'literacy':"Mother's literacy",'anc_visits':'Antenatal visits','anc_timing':'Timing of first antenatal visit','delivery_facility':'Place of delivery','deworming':'Deworming','v190':'Wealth quintile','v025':'Residence','v024':'Region','water_source_grp':'Water source','water_time':'Time to water source','toilet_grp':'Toilet type','shared_toilet':'Shared toilet','v481':'Covered by health insurance (DHS v481)','distance_problem':'Distance to health facility a problem','v012':"Mother's age",'v136':'Household size','v137':'Children under five in household','v404':'Mother currently breastfeeding (DHS v404)','iron_supp':'Iron supplementation','m17':'Caesarean delivery','birth_size':'Size at birth'}
def load(f):
    p = os.path.join(HERE, f)
    if not os.path.exists(p): raise FileNotFoundError(f"Cannot find {f} in {HERE}")
    d = pd.read_stata(p, columns=ml_features)
    for v in categorical:
        s = d[v].astype(object).where(d[v].notna(), None)
        d[v] = s.map(lambda z: None if (z is None or str(z).strip() in ('', 'nan', '.')) else str(z).strip().lower())
    for v in ml_features:
        if v not in categorical: d[v] = pd.to_numeric(d[v], errors='coerce')
    return d
D = {'2019-20': load(F20), '2024-25': load(F25)}
rows = []
for v in ml_features:
    row = {'Predictor': v, 'Description': desc.get(v, ''), 'Type': 'category' if v in categorical else 'number',
           'Dropped in harmonised run': 'yes' if v in dropped else ''}
    for lab, d in D.items():
        x = d[v]; row[f'Missing % {lab}'] = round(100 * x.isna().mean(), 1)
        if v in categorical:
            vc = x.dropna().value_counts(normalize=True).mul(100).round(1)
            row[f'Summary {lab}'] = '; '.join(f'{k} {p}%' for k, p in vc.items())
        else:
            row[f'Summary {lab}'] = f'{x.mean():.2f} ({x.std():.2f})'
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, 'predictor_table_by_survey.csv'), index=False)
with pd.option_context('display.width', 250, 'display.max_columns', 20, 'display.max_colwidth', 60):
    print(out[['Predictor', 'Type', 'Missing % 2019-20', 'Missing % 2024-25', 'Dropped in harmonised run']].to_string(index=False))
print("\nWrote predictor_table_by_survey.csv (aggregate only).")
