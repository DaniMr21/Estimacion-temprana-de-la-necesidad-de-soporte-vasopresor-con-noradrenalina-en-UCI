import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

#CONFIGURACIÓN
RUTA_CSV    = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'
ETIQUETA    = 'etiqueta_norad_12_48'

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

variables_predictoras = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'rr_max',             # sin fio2_max
    'ventilacion_invasiva_12h', 'gcs_min',       # 12h
    'creatinina_max', 'diuresis_ml_kg_12h',      # 12h
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]

#CARGA Y FILTRADO
print("─" * 60)
print("REGRESIÓN LOGÍSTICA MULTIVARIANTE — v4l LARGA (25 variables)")
print("─" * 60)

df = pd.read_csv(RUTA_CSV)
df = df.dropna(subset=['pf_max'])
print(f"\nEstancias totales      : {len(df)}")
df = (df.sort_values(['subject_id', 'contador_estancia_uci'])
        .drop_duplicates('subject_id', keep='first')
        .reset_index(drop=True))
print(f"Tras filtrar 1ª estancia: {len(df)}")
print(f"Positivos: {df[ETIQUETA].sum()} ({100*df[ETIQUETA].mean():.2f}%)\n")

#PREPARACIÓN
X = df[variables_predictoras].copy()
X['gender'] = (X['gender'] == 'M').astype(int)
y = df[ETIQUETA].astype(int)


X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

#AJUSTE
print("[1/3] Ajustando modelo...")
X_sm = sm.add_constant(X_std)
modelo = sm.Logit(y, X_sm).fit(disp=0)

print(f"  Pseudo R²     : {modelo.prsquared:.4f}")
print(f"  LLR p-value   : {modelo.llr_pvalue:.2e}")
print(f"  AIC           : {modelo.aic:.2f}")
print(f"  Convergencia  : {'OK' if modelo.mle_retvals['converged'] else 'NO CONVERGIÓ'}\n")

#TABLA
ic = modelo.conf_int(); ic.columns = ['IC_inf', 'IC_sup']
tabla = pd.DataFrame({
    'variable':    modelo.params.index,
    'coef':        modelo.params.values,
    'IC_inf_coef': ic['IC_inf'].values,
    'IC_sup_coef': ic['IC_sup'].values,
    'OR':          np.exp(modelo.params.values),
    'OR_IC_inf':   np.exp(ic['IC_inf'].values),
    'OR_IC_sup':   np.exp(ic['IC_sup'].values),
    'p_valor':     modelo.pvalues.values,
})
tabla_vars = tabla[tabla['variable'] != 'const'].copy().reset_index(drop=True)

rechaza, p_corr, _, _ = multipletests(tabla_vars['p_valor'], method='fdr_bh', alpha=0.05)
tabla_vars['p_valor_BH']          = p_corr
tabla_vars['significativa_BH']    = rechaza
tabla_vars['significativa_bruta'] = tabla_vars['p_valor'] < 0.05
tabla_vars = tabla_vars.sort_values('p_valor').reset_index(drop=True)

def fmt_p(p):
    return f'{p:.2e}' if p < 1e-4 else f'{p:.4f}'

tp = tabla_vars[['variable','coef','OR','OR_IC_inf','OR_IC_sup',
                 'p_valor','p_valor_BH','significativa_BH']].copy()
for col in ['coef','OR','OR_IC_inf','OR_IC_sup']:
    tp[col] = tp[col].round(3)
tp['p_valor']    = tp['p_valor'].apply(fmt_p)
tp['p_valor_BH'] = tp['p_valor_BH'].apply(fmt_p)

print("[2/3] Tabla de coeficientes:\n")
print(tp.to_string(index=False))

n_sig = tabla_vars['significativa_BH'].sum()
print(f"\nSignificativas tras BH-FDR: {n_sig} / {len(tabla_vars)}")
print("Significativas:")
for _, f in tabla_vars[tabla_vars['significativa_BH']].iterrows():
    d = "↑ riesgo" if f['coef'] > 0 else "↓ riesgo"
    print(f"  - {f['variable']:<28} OR={f['OR']:.2f} [{f['OR_IC_inf']:.2f}, {f['OR_IC_sup']:.2f}]  ({d})")
print("NO significativas:")
for v in tabla_vars[~tabla_vars['significativa_BH']]['variable']:
    print(f"  - {v}")

#COMPARACIÓN CON UNIVARIANTE
print("\n[3/3] Comparación con univariante...")
ruta_uni = os.path.join(CARPETA_TABLAS, 'significancia_univariante_v4l.csv')
if os.path.exists(ruta_uni):
    uni = pd.read_csv(ruta_uni)[['variable','p_valor_BH','significativa_BH']]
    uni.columns = ['variable','p_uni_BH','sig_uni_BH']
    multi = tabla_vars[['variable','p_valor_BH','significativa_BH','coef','OR']].copy()
    multi.columns = ['variable','p_multi_BH','sig_multi_BH','coef_multi','OR_multi']
    comp = uni.merge(multi, on='variable', how='outer')

    def cat(row):
        if row['sig_uni_BH'] and row['sig_multi_BH']:    return 'Sig. en ambos (robusta)'
        if row['sig_uni_BH'] and not row['sig_multi_BH']:return 'Solo univariante (absorbida)'
        if not row['sig_uni_BH'] and row['sig_multi_BH']:return 'Solo multivariante (efecto ajustado)'
        return 'No sig. en ninguno'

    comp['categoria'] = comp.apply(cat, axis=1)
    comp = comp.sort_values('p_multi_BH').reset_index(drop=True)

    for c in ['Sig. en ambos (robusta)', 'Solo univariante (absorbida)',
              'Solo multivariante (efecto ajustado)', 'No sig. en ninguno']:
        vs = comp[comp['categoria'] == c]['variable'].tolist()
        print(f"\n  [{c}] ({len(vs)}):")
        for v in vs: print(f"    - {v}")

    comp.to_csv(os.path.join(CARPETA_TABLAS,
                'comparacion_univariante_vs_multivariante_v4l.csv'), index=False)
else:
    print(f"  No encontrado: {ruta_uni}. Ejecuta primero significancia_univariante_v4l.py")

tabla_vars.to_csv(os.path.join(CARPETA_TABLAS,
                 'regresion_logistica_multivariante_v4l.csv'), index=False)
print(f"\nTabla guardada en: tablas/regresion_logistica_multivariante_v4l.csv")
