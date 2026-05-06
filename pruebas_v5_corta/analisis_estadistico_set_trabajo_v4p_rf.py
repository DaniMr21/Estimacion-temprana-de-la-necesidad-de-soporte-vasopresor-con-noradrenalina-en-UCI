import warnings
warnings.filterwarnings('ignore')

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'
ETIQUETA = 'etiqueta_norad_3_12'
N_PERM   = 50
RS       = 42

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
CARPETA_FIGS   = os.path.join(CARPETA_BASE, 'figuras')

os.makedirs(CARPETA_TABLAS, exist_ok=True)
os.makedirs(CARPETA_FIGS, exist_ok=True)


VARIABLES = ['map_min', 'hr_media', 'pf_min', 'spo2_min', 'rr_max', 
             'diuresis_ml_kg_3h', 'lactato_max', 'ph_min', 'bilirrubina_media',
             'temp_min', 'sofa_max', 'creatinina_max', 'plaquetas_min', 'tp_max', 'glucemia_min']

VARS_BINARIAS = []
VARS_CONTINUAS = [v for v in VARIABLES if v not in VARS_BINARIAS]


def fmt_p(p):
    if pd.isna(p):
        return 'NaN'

    return f'{p:.2e}' if p < 1e-4 else f'{p:.4f}'


# ── CARGA ─────────────────────────────────────────────────────────────────────
print("=" * 68)
print("ANÁLISIS ESTADÍSTICO SET TRABAJO — v4p CORTA RF")
print("=" * 68)

df_full = pd.read_csv(RUTA_CSV)
df_full = df_full.dropna(subset=['pf_max'])


df_1 = (
    df_full
    .sort_values(['subject_id', 'contador_estancia_uci'])
    .drop_duplicates('subject_id', keep='first')
    .reset_index(drop=True)
)

print(f"\nDataset (1ª estancia/paciente): N={len(df_1)} | "
      f"Positivos={df_1[ETIQUETA].sum()} "
      f"({100*df_1[ETIQUETA].mean():.2f}%)")


X_full = df_full[VARIABLES].copy()
y_full = df_full[ETIQUETA].copy()
pid_full = df_full['subject_id'].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UNIVARIANTE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*68)
print("[1/3] SIGNIFICANCIA UNIVARIANTE")
print("─"*68)

X1 = df_1[VARIABLES].copy()
y1 = df_1[ETIQUETA].copy()

filas = []

for var in VARS_CONTINUAS:
    pos = X1.loc[y1 == 1, var].dropna()
    neg = X1.loc[y1 == 0, var].dropna()

    if len(pos) < 2 or len(neg) < 2:
        filas.append({
            'variable': var,
            'test': 'Mann-Whitney U',
            'mediana_pos': np.nan,
            'mediana_neg': np.nan,
            'p_valor': np.nan
        })
        continue

    stat, p = mannwhitneyu(pos, neg, alternative='two-sided')

    filas.append({
        'variable': var,
        'test': 'Mann-Whitney U',
        'mediana_pos': round(pos.median(), 4),
        'mediana_neg': round(neg.median(), 4),
        'p_valor': p
    })


for var in VARS_BINARIAS:
    tabla = pd.crosstab(X1[var], y1)

    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        filas.append({
            'variable': var,
            'test': 'Chi-cuadrado',
            'mediana_pos': np.nan,
            'mediana_neg': np.nan,
            'p_valor': np.nan
        })
        continue

    chi2, p, _, _ = chi2_contingency(tabla)

    filas.append({
        'variable': var,
        'test': 'Chi-cuadrado',
        'mediana_pos': round(X1.loc[y1 == 1, var].mean(), 4),
        'mediana_neg': round(X1.loc[y1 == 0, var].mean(), 4),
        'p_valor': p
    })


uni = pd.DataFrame(filas)

mask = uni['p_valor'].notna()
rechaza, p_corr, _, _ = multipletests(
    uni.loc[mask, 'p_valor'],
    method='fdr_bh',
    alpha=0.05
)

uni['p_valor_BH'] = np.nan
uni['sig_BH'] = False

uni.loc[mask, 'p_valor_BH'] = p_corr
uni.loc[mask, 'sig_BH'] = rechaza

uni = uni.sort_values('p_valor').reset_index(drop=True)

rp = uni.copy()
rp['p_valor'] = rp['p_valor'].apply(fmt_p)
rp['p_valor_BH'] = rp['p_valor_BH'].apply(fmt_p)

print(rp.to_string(index=False))
print(f"\n  Significativas BH-FDR: {uni['sig_BH'].sum()} / {len(uni)}")

uni.to_csv(
    os.path.join(CARPETA_TABLAS, 'univariante_set_trabajo_v4p_rf.csv'),
    index=False
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REGRESIÓN LOGÍSTICA MULTIVARIANTE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*68)
print("[2/3] REGRESIÓN LOGÍSTICA MULTIVARIANTE")
print("─"*68)

X1_imp = X1.fillna(X1.median(numeric_only=True))

X1_std = pd.DataFrame(
    StandardScaler().fit_transform(X1_imp),
    columns=X1_imp.columns,
    index=X1_imp.index
)

X_sm = sm.add_constant(X1_std)
modelo_lr = sm.Logit(y1, X_sm).fit(disp=0)

print(f"  Pseudo R²: {modelo_lr.prsquared:.4f}  |  "
      f"LLR p: {modelo_lr.llr_pvalue:.2e}  |  "
      f"AIC: {modelo_lr.aic:.2f}  |  "
      f"Convergencia: {'OK' if modelo_lr.mle_retvals['converged'] else 'NO'}")

ic = modelo_lr.conf_int()
ic.columns = ['IC_inf', 'IC_sup']

multi = pd.DataFrame({
    'variable': modelo_lr.params.index,
    'coef': modelo_lr.params.values.round(3),
    'OR': np.exp(modelo_lr.params.values).round(3),
    'OR_IC_inf': np.exp(ic['IC_inf'].values).round(3),
    'OR_IC_sup': np.exp(ic['IC_sup'].values).round(3),
    'p_valor': modelo_lr.pvalues.values,
})

multi = multi[multi['variable'] != 'const'].reset_index(drop=True)

rechaza2, p_corr2, _, _ = multipletests(
    multi['p_valor'],
    method='fdr_bh',
    alpha=0.05
)

multi['p_valor_BH'] = p_corr2
multi['sig_BH'] = rechaza2

multi = multi.sort_values('p_valor').reset_index(drop=True)

mp = multi[[
    'variable', 'coef', 'OR', 'OR_IC_inf',
    'OR_IC_sup', 'p_valor', 'p_valor_BH', 'sig_BH'
]].copy()

mp['p_valor'] = mp['p_valor'].apply(fmt_p)
mp['p_valor_BH'] = mp['p_valor_BH'].apply(fmt_p)

print("\n" + mp.to_string(index=False))
print(f"\n  Significativas BH-FDR: {multi['sig_BH'].sum()} / {len(multi)}")

multi.to_csv(
    os.path.join(CARPETA_TABLAS, 'multivariante_set_trabajo_v4p_rf.csv'),
    index=False
)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PERMUTATION IMPORTANCE — RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─"*68)
print("[3/3] PERMUTATION IMPORTANCE — RANDOM FOREST")
print("─"*68)

pipeline_pi = Pipeline([
    ('modelo', RandomForestClassifier(random_state=RS, n_jobs=-1))
])

espacio_pi = {
    'modelo__n_estimators':     [300, 500, 1000],
    'modelo__max_depth':        [10, 20, None],
    'modelo__min_samples_leaf': [1, 5],
    'modelo__max_features':     ['sqrt'],
    'modelo__class_weight':     ['balanced', 'balanced_subsample'],
}

cv_ext = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
cv_int = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RS)

nombres = VARIABLES
n_vars = len(nombres)

imp_medias = np.zeros((5, n_vars))
todas_perm = []
aucs_pi = []

t_pi = time.time()

for nf, (idx_tr, idx_te) in enumerate(
    cv_ext.split(X_full, y_full, groups=pid_full),
    start=1
):

    x_tr, x_te = X_full.iloc[idx_tr], X_full.iloc[idx_te]
    y_tr, y_te = y_full.iloc[idx_tr], y_full.iloc[idx_te]
    pid_tr = pid_full.iloc[idx_tr]

    gs = GridSearchCV(
        pipeline_pi,
        espacio_pi,
        cv=cv_int,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )

    gs.fit(x_tr, y_tr, groups=pid_tr)

    proba = gs.predict_proba(x_te)[:, 1]
    auc = roc_auc_score(y_te, proba)

    aucs_pi.append(auc)

    res = permutation_importance(
        gs.best_estimator_,
        x_te,
        y_te,
        scoring='roc_auc',
        n_repeats=N_PERM,
        random_state=RS,
        n_jobs=-1
    )

    imp_medias[nf - 1, :] = res.importances_mean
    todas_perm.append(res.importances)

    print(f"  Fold {nf}: AUC={auc:.4f}")


auc_pi = np.mean(aucs_pi)
std_pi = np.std(aucs_pi)

print(f"\n  AUC medio RF ({len(VARIABLES)} vars): {auc_pi:.4f} ± {std_pi:.4f}  "
      f"({(time.time()-t_pi)/60:.1f} min)")


filas_pi = []

for i, nom in enumerate(nombres):
    vals = imp_medias[:, i]

    media = vals.mean()
    desv = vals.std()

    ic_inf = np.percentile(vals, 2.5)
    ic_sup = np.percentile(vals, 97.5)

    perm_c = np.concatenate([m[i, :] for m in todas_perm])
    p_emp = np.mean(perm_c <= 0)

    filas_pi.append({
        'variable': nom,
        'caida_AUC_pp': media * 100,
        'caida_AUC_desv': desv,
        'IC_inf': ic_inf,
        'IC_sup': ic_sup,
        'p_valor_empirico': p_emp,
        'IC_excluye_cero': (ic_inf > 0),
        'p_emp_sig': (p_emp < 0.05)
    })


pi = (
    pd.DataFrame(filas_pi)
    .sort_values('caida_AUC_pp', ascending=False)
    .reset_index(drop=True)
)


tp = pi.copy()

for c in [
    'caida_AUC_pp', 'caida_AUC_desv',
    'IC_inf', 'IC_sup', 'p_valor_empirico'
]:
    tp[c] = tp[c].round(4)

print("\n" + tp.to_string(index=False))
print(f"\n  Importantes (IC95% excluye 0): {pi['IC_excluye_cero'].sum()} / {len(pi)}")


# ── FIGURA ────────────────────────────────────────────────────────────────────
df_plot = pi.sort_values('caida_AUC_pp').copy()

colores = [
    '#2ca02c' if r else '#d62728'
    for r in df_plot['IC_excluye_cero']
]

fig, ax = plt.subplots(figsize=(10, 7))

pos = np.arange(len(df_plot))

ax.barh(
    pos,
    df_plot['caida_AUC_pp'],
    xerr=df_plot['caida_AUC_desv'] * 100,
    color=colores,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.85,
    capsize=3
)

ax.set_yticks(pos)
ax.set_yticklabels(df_plot['variable'])

ax.set_xlabel('Caída de AUC (pp) al permutar la variable')

ax.set_title(
    f'Permutation Importance — RF SET TRABAJO v4p CORTA ({len(VARIABLES)} vars)\n'
    f'AUC = {auc_pi:.4f} | 5 folds | {N_PERM} perm/fold'
)

ax.axvline(0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

ax.legend(handles=[
    Patch(facecolor='#2ca02c', edgecolor='black', label='IC95% excluye 0'),
    Patch(facecolor='#d62728', edgecolor='black', label='IC95% incluye 0'),
], loc='lower right')

plt.tight_layout()

ruta_fig = os.path.join(
    CARPETA_FIGS,
    'permutation_importance_set_trabajo_v4p_rf.png'
)

plt.savefig(ruta_fig, dpi=150, bbox_inches='tight')
plt.close()


pi.to_csv(
    os.path.join(CARPETA_TABLAS, 'permutation_importance_set_trabajo_v4p_rf.csv'),
    index=False
)

print(f"\n  Figura: {ruta_fig}")

print("\n" + "═"*68)
print("FIN — v4p CORTA RF")
print("═"*68)
