import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# 1. CARGA

RUTA = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'
df = pd.read_csv(RUTA)
df = df.dropna(subset=['pf_max'])

x = df['tiene_sepsis'].astype(int).values
y = df['etiqueta_norad_3_12'].astype(int).values
grupos = df['subject_id'].values

n_total = len(df)
prevalencia_evento = 100 * y.mean()
prevalencia_sepsis = 100 * x.mean()

print("=" * 70)
print("CAPACIDAD PREDICTIVA DE `tiene_sepsis` vs `etiqueta_norad_3_12`")
print("Ventana PRECOZ (observación 0-3h, predicción 3-12h)")
print("=" * 70)
print(f"N estancias                 : {n_total}")
print(f"Prevalencia norad 3-12h     : {prevalencia_evento:.2f}%")
print(f"Prevalencia sepsis en 0-3h  : {prevalencia_sepsis:.2f}%")
print()

# 2. TABLA DE CONTINGENCIA

tabla = pd.crosstab(df['tiene_sepsis'], df['etiqueta_norad_3_12'],
                    rownames=['tiene_sepsis'], colnames=['etiqueta_norad_3_12'])
print("TABLA DE CONTINGENCIA 2x2")
print(tabla)
print()

a = int(tabla.loc[1, 1])  # sepsis=1, evento=1
b = int(tabla.loc[1, 0])  # sepsis=1, evento=0
c = int(tabla.loc[0, 1])  # sepsis=0, evento=1
d = int(tabla.loc[0, 0])  # sepsis=0, evento=0


# 3. RIESGO RELATIVO Y ODDS RATIO (con IC 95%)

riesgo_expuestos = a / (a + b)
riesgo_no_exp    = c / (c + d)
rr = riesgo_expuestos / riesgo_no_exp

log_rr = np.log(rr)
se_log_rr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
ic_rr = (np.exp(log_rr - 1.96 * se_log_rr),
         np.exp(log_rr + 1.96 * se_log_rr))

odds_exp    = a / b
odds_no_exp = c / d
or_ = odds_exp / odds_no_exp
log_or = np.log(or_)
se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
ic_or = (np.exp(log_or - 1.96 * se_log_or),
         np.exp(log_or + 1.96 * se_log_or))

print("RIESGO POR GRUPO")
print(f"  P(norad | sepsis)    = {riesgo_expuestos:.4f}  ({100*riesgo_expuestos:.2f}%)")
print(f"  P(norad | no sepsis) = {riesgo_no_exp:.4f}  ({100*riesgo_no_exp:.2f}%)")
print()
print(f"  Riesgo relativo (RR) = {rr:.2f}  IC95% [{ic_rr[0]:.2f}, {ic_rr[1]:.2f}]")
print(f"  Odds ratio     (OR)  = {or_:.2f}  IC95% [{ic_or[0]:.2f}, {ic_or[1]:.2f}]")
print()


# 4. SENSIBILIDAD / ESPECIFICIDAD

vp = a
fn = c
fp = b
vn = d

sensibilidad = vp / (vp + fn)
especificidad = vn / (vn + fp)
vpp = vp / (vp + fp)
vpn = vn / (vn + fn)

print("SEPSIS COMO CLASIFICADOR BINARIO CRUDO (umbral trivial = 1)")
print(f"  Sensibilidad          = {sensibilidad:.4f}  ({100*sensibilidad:.1f}%)")
print(f"  Especificidad         = {especificidad:.4f}  ({100*especificidad:.1f}%)")
print(f"  Valor predictivo pos. = {vpp:.4f}  ({100*vpp:.1f}%)")
print(f"  Valor predictivo neg. = {vpn:.4f}  ({100*vpn:.1f}%)")
print()

# 5. AUC Y 6. INFORMACIÓN MUTUA

auc = roc_auc_score(y, x)
print("DISCRIMINACIÓN")
print(f"  AUC (sepsis sola)     = {auc:.4f}")
print(f"  (0.50 = azar; 1.00 = perfecto)")
print()

mi = mutual_info_classif(x.reshape(-1, 1), y,
                         discrete_features=True, random_state=42)[0]
print(f"  Información mutua     = {mi:.5f} nats")
print()

# 7. TEST CHI-CUADRADO DE INDEPENDENCIA

chi2, p_valor, gl, esperados = stats.chi2_contingency(tabla)
print("TEST DE INDEPENDENCIA (chi-cuadrado)")
print(f"  chi2 = {chi2:.2f}  gl = {gl}  p = {p_valor:.4g}")
print(f"  H0: sepsis y norad son independientes. "
      f"{'Se rechaza' if p_valor < 0.05 else 'No se rechaza'} al 5%.")
print()


# 8. GANANCIA MARGINAL EN EL MODELO

variables_todas = [
    'anchor_age', 'gender', 'contador_estancia_uci',
    'sofa_media', 'sofa_min', 'sofa_max',
    'ventilacion_invasiva_3h',
    'lactato_media', 'lactato_min', 'lactato_max',
    'creatinina_media', 'creatinina_min', 'creatinina_max',
    'plaquetas_media', 'plaquetas_min', 'plaquetas_max',
    'bilirrubina_media', 'bilirrubina_min', 'bilirrubina_max',
    'tp_media', 'tp_min', 'tp_max',
    'gpt_media', 'gpt_min', 'gpt_max',
    'got_media', 'got_min', 'got_max',
    'pao2_media', 'pao2_min', 'pao2_max',
    'ph_media', 'ph_min', 'ph_max',
    'leucocitos_media', 'leucocitos_min', 'leucocitos_max',
    'paco2_media', 'paco2_min', 'paco2_max',
    'bicarbonato_media', 'bicarbonato_min', 'bicarbonato_max',
    'glucemia_media', 'glucemia_min', 'glucemia_max',
    'hemoglobina_media', 'hemoglobina_min', 'hemoglobina_max',
    'hr_media', 'hr_min', 'hr_max',
    'rr_media', 'rr_min', 'rr_max',
    'temp_media', 'temp_min', 'temp_max',
    'spo2_media', 'spo2_min', 'spo2_max',
    'map_media', 'map_min', 'map_max',
    'fio2_media', 'fio2_min', 'fio2_max',
    'pf_media', 'pf_min', 'pf_max',
    'gcs_media', 'gcs_min', 'gcs_max',
    'peso_kg',
    'diuresis_ml_kg_3h',
]

df_mod = df.copy()
df_mod['gender'] = (df_mod['gender'] == 'M').astype(int)

x_sin = df_mod[variables_todas].values
x_con = df_mod[variables_todas + ['tiene_sepsis']].values
y_mod = df_mod['etiqueta_norad_3_12'].values
grupos_mod = df_mod['subject_id'].values

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ('escalador', RobustScaler()),
    ('modelo', LogisticRegression(max_iter=5000, class_weight='balanced',
                                  solver='liblinear', random_state=42, C=0.01))
])

print("GANANCIA MARGINAL EN EL MODELO (regresión logística, CV 5-fold agrupada)")
print("  Entrenando baseline SIN tiene_sepsis...")
auc_sin = cross_val_score(pipe, x_sin, y_mod, groups=grupos_mod,
                          cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"    AUC sin sepsis : {auc_sin.mean():.4f} +/- {auc_sin.std():.4f}")

print("  Entrenando baseline CON tiene_sepsis...")
auc_con = cross_val_score(pipe, x_con, y_mod, groups=grupos_mod,
                          cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"    AUC con sepsis : {auc_con.mean():.4f} +/- {auc_con.std():.4f}")

delta = auc_con.mean() - auc_sin.mean()
print(f"    Delta AUC      : {delta:+.4f}")
if delta > 0.005:
    print("    -> la variable APORTA de forma apreciable al modelo.")
elif delta > 0:
    print("    -> la variable aporta algo, pero marginalmente.")
else:
    print("    -> la variable NO aporta o incluso resta (posible redundancia).")
print()


# RESUMEN EJECUTIVO

print("-----------------")
print("RESUMEN (ventana PRECOZ 0-3h / 3-12h)")
print("----------------")
print(f"  RR (sepsis vs no sepsis)   : {rr:.2f}  IC95% [{ic_rr[0]:.2f}, {ic_rr[1]:.2f}]")
print(f"  OR                         : {or_:.2f}  IC95% [{ic_or[0]:.2f}, {ic_or[1]:.2f}]")
print(f"  p-valor chi2               : {p_valor:.4g}")
print(f"  AUC como predictor unico   : {auc:.4f}")
print(f"  Informacion mutua          : {mi:.5f}")
print(f"  Delta AUC en modelo LR     : {delta:+.4f}")
