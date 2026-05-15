import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')
 
# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
RUTA_CSV   = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
RUTA_PKL   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELOS_ENTRENADOS\modelo_Medio_6_24_XGB.pkl')
RUTA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELOS_ENTRENADOS\modelo_Medio_6_24_XGB_calibrado.pkl')
 
ETIQUETA   = 'etiqueta_norad_6_24'
COLUMNA_ID = 'subject_id'
VARIABLES  = ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media', 'sofa_max', 'ventilacion_invasiva_6h']
N_SPLITS   = 5
RANDOM_SEED = 42
 
 
# ── FUNCIONES MÉTRICAS ─────────────────────────────────────────────────────────
 
def calcular_ece(probabilidades, etiquetas, n_bins=10):
    limites = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(etiquetas)
    for i in range(n_bins):
        mascara = (probabilidades >= limites[i]) & (probabilidades < limites[i + 1])
        n_bin = mascara.sum()
        if n_bin == 0:
            continue
        ece += (n_bin / n_total) * abs(probabilidades[mascara].mean() - etiquetas[mascara].mean())
    return float(ece)
 
 
def calcular_bss(probabilidades, etiquetas):
    prevalencia     = etiquetas.mean()
    bs_modelo       = brier_score_loss(etiquetas, probabilidades)
    bs_climatologia = brier_score_loss(etiquetas, np.full_like(probabilidades, prevalencia))
    return float(1 - bs_modelo / bs_climatologia) if bs_climatologia != 0 else np.nan
 
 
# ── CARGA ──────────────────────────────────────────────────────────────────────
print("Cargando datos y modelo...")
df     = pd.read_csv(RUTA_CSV).dropna(subset=['pf_max'])
y      = df[ETIQUETA].values.astype(int)
grupos = df[COLUMNA_ID].values
X      = df[VARIABLES].copy()
modelo = joblib.load(RUTA_PKL)
 
print(f"  Pacientes: {len(y)} | Positivos: {y.sum()} ({y.mean():.3f})")
 
# ── PASO 1: PROBABILIDADES OOF HONESTAS ───────────────────────────────────────
print(f"\nGenerando probabilidades OOF ({N_SPLITS} folds)...")
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
 
prob_oof          = np.zeros(len(y))
prob_platt_puro   = np.zeros(len(y))   # C=1e10 — sigmoide libre, sin regularización
prob_platt_reg    = np.zeros(len(y))   # C óptimo por CV — regularización automática
prob_isoton       = np.zeros(len(y))   # Isotónica — forma libre, no paramétrica
 
for fold_idx, (idx_train, idx_test) in enumerate(cv.split(X, y, grupos)):
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train         = y[idx_train]
 
    modelo_fold = clone(modelo)
    modelo_fold.fit(X_train, y_train)
 
    prob_bruta_train            = modelo_fold.predict_proba(X_train)[:, 1]
    prob_bruta_test             = modelo_fold.predict_proba(X_test)[:, 1]
    prob_oof[idx_test]          = prob_bruta_test
 
    # Platt puro (sigmoide, sin regularización)
    cal_platt_puro = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    cal_platt_puro.fit(prob_bruta_train.reshape(-1, 1), y_train)
    prob_platt_puro[idx_test] = cal_platt_puro.predict_proba(
        prob_bruta_test.reshape(-1, 1))[:, 1]
 
    # Platt regularizado (C óptimo por CV interna en cada fold)
    cal_platt_reg = LogisticRegressionCV(
        Cs=10, cv=3, solver='lbfgs', max_iter=1000, scoring='neg_brier_score'
    )
    cal_platt_reg.fit(prob_bruta_train.reshape(-1, 1), y_train)
    prob_platt_reg[idx_test] = cal_platt_reg.predict_proba(
        prob_bruta_test.reshape(-1, 1))[:, 1]
 
    # Isotónica (no paramétrica — más flexible pero necesita más datos)
    cal_iso = IsotonicRegression(out_of_bounds='clip')
    cal_iso.fit(prob_bruta_train, y_train)
    prob_isoton[idx_test] = cal_iso.predict(prob_bruta_test)
 
    print(f"  Fold {fold_idx + 1}/{N_SPLITS} — test n={len(idx_test)}, positivos={y[idx_test].sum()}")
 
# ── PASO 2: COMPARACIÓN DE CALIBRADORES ───────────────────────────────────────
print(f"\n{'Calibrador':<20} {'AUC':>7} {'BSS':>8} {'ECE':>8}")
print("-" * 47)
candidatos = {
    'Sin calibrar'  : prob_oof,
    'Platt puro'    : prob_platt_puro,
    'Platt reg (CV)': prob_platt_reg,
    'Isotónica'     : prob_isoton,
}
metricas_candidatos = {}
for nombre, prob in candidatos.items():
    auc = roc_auc_score(y, prob)
    bss = calcular_bss(prob, y)
    ece = calcular_ece(prob, y)
    metricas_candidatos[nombre] = {'auc': auc, 'bss': bss, 'ece': ece}
    print(f"  {nombre:<18} {auc:>7.4f} {bss:>+8.4f} {ece:>8.4f}")
 
# Elegir el mejor calibrador por ECE (con BSS > 0 como condición necesaria)
candidatos_validos = {n: m for n, m in metricas_candidatos.items()
                      if m['bss'] > 0 and n != 'Sin calibrar'}
if candidatos_validos:
    mejor_nombre = min(candidatos_validos, key=lambda n: candidatos_validos[n]['ece'])
else:
    mejor_nombre = 'Sin calibrar'
    print("\n  AVISO: ningún calibrador mantiene BSS > 0")
 
print(f"\n  → Mejor calibrador: {mejor_nombre}")
prob_calibrada = candidatos[mejor_nombre]
 
# ── PASO 3: WRAPPER Y GUARDADO ─────────────────────────────────────────────────
 
class ModeloCalibrado:
    """
    Wrapper genérico: modelo XGB + calibrador (Platt puro, Platt reg o Isotónica).
    predict_proba() es compatible con sklearn.
    El modelo base se re-entrena sobre TODOS los datos internos antes de guardar.
    """
    def __init__(self, modelo_base, calibrador, tipo_calibrador):
        self.modelo_base      = modelo_base
        self.calibrador       = calibrador
        self.tipo_calibrador  = tipo_calibrador
 
    def predict_proba(self, X):
        prob_bruta = self.modelo_base.predict_proba(X)[:, 1]
        if isinstance(self.calibrador, IsotonicRegression):
            prob_cal = self.calibrador.predict(prob_bruta)
        else:
            prob_cal = self.calibrador.predict_proba(prob_bruta.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - prob_cal, prob_cal])
 
    def predict(self, X, umbral=0.5):
        return (self.predict_proba(X)[:, 1] >= umbral).astype(int)
 
 
# Ajustar el calibrador elegido sobre las probabilidades OOF completas
print(f"\nAjustando '{mejor_nombre}' sobre probabilidades OOF completas...")
if mejor_nombre == 'Platt puro':
    calibrador_final = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrador_final.fit(prob_oof.reshape(-1, 1), y)
elif mejor_nombre == 'Platt reg (CV)':
    calibrador_final = LogisticRegressionCV(
        Cs=10, cv=3, solver='lbfgs', max_iter=1000, scoring='neg_brier_score'
    )
    calibrador_final.fit(prob_oof.reshape(-1, 1), y)
elif mejor_nombre == 'Isotónica':
    calibrador_final = IsotonicRegression(out_of_bounds='clip')
    calibrador_final.fit(prob_oof, y)
else:
    calibrador_final = None
 
# Re-entrenar el modelo base sobre TODOS los datos internos
print("Re-entrenando modelo base sobre datos internos completos...")
modelo_completo = clone(modelo)
modelo_completo.fit(X, y)
 
modelo_calibrado = ModeloCalibrado(modelo_completo, calibrador_final, mejor_nombre)
joblib.dump(modelo_calibrado, RUTA_SALIDA)
print(f"Modelo calibrado ({mejor_nombre}) guardado en: {RUTA_SALIDA}")
 
# ── PASO 4: GRÁFICA DE VERIFICACIÓN ───────────────────────────────────────────
colores_graf = {
    'Sin calibrar'  : '#ff7f0e',
    'Platt puro'    : '#1f77b4',
    'Platt reg (CV)': '#9467bd',
    'Isotónica'     : '#2ca02c',
}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Calibración {mejor_nombre} — XGB Medio_6_24', fontsize=13, fontweight='bold')
 
# Curvas de calibración — todos los candidatos
for nombre, prob in candidatos.items():
    frac_pos, media_pred = calibration_curve(y, prob, n_bins=10, strategy='quantile')
    lw = 2.5 if nombre in ('Sin calibrar', mejor_nombre) else 1.2
    alpha = 1.0 if nombre in ('Sin calibrar', mejor_nombre) else 0.45
    axes[0].plot(media_pred, frac_pos, marker='o', lw=lw, alpha=alpha,
                 color=colores_graf[nombre], label=nombre, markersize=5)
 
axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfecta')
prob_max = max(p.max() for p in candidatos.values())
axes[0].set_xlim(-0.01, min(prob_max + 0.05, 1.0))
axes[0].set_xlabel('Probabilidad predicha media')
axes[0].set_ylabel('Fracción de positivos')
axes[0].set_title('Curvas de calibración (bins por cuantil)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.4)
 
# Histograma sin calibrar vs elegido
axes[1].hist(prob_oof,                bins=30, alpha=0.6,
             color=colores_graf['Sin calibrar'], label='Sin calibrar', density=True)
axes[1].hist(candidatos[mejor_nombre], bins=30, alpha=0.6,
             color=colores_graf[mejor_nombre],   label=mejor_nombre,   density=True)
axes[1].set_xlabel('Probabilidad predicha')
axes[1].set_ylabel('Densidad')
axes[1].set_title('Distribución de probabilidades')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.4)
 
plt.tight_layout()
ruta_fig = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'verificacion_calibracion_medio_xgb.png')
plt.savefig(ruta_fig, dpi=200, bbox_inches='tight')
plt.show()
print(f"\nGráfica guardada en: {ruta_fig}")
print("\n[Fin] El modelo calibrado está listo para validación externa.")

# ── PASO 5: AÑADIR RESULTADOS AL CSV GENERAL ──────────────────────────────────
from sklearn.metrics import average_precision_score, log_loss

print("\nCalculando métricas por fold para el CSV...")

# 1. Calcular métricas por fold para tener las medias y desviaciones (std)
metricas_cal = {'AUC_ROC': [], 'AUC_PR': [], 'Brier': [], 'BSS': [], 'ECE': [], 'LogLoss': []}

for fold_idx, (idx_train, idx_test) in enumerate(cv.split(X, y, grupos)):
    y_test_fold = y[idx_test]
    prob_fold = prob_calibrada[idx_test]
    
    metricas_cal['AUC_ROC'].append(roc_auc_score(y_test_fold, prob_fold))
    metricas_cal['AUC_PR'].append(average_precision_score(y_test_fold, prob_fold))
    metricas_cal['Brier'].append(brier_score_loss(y_test_fold, prob_fold))
    metricas_cal['BSS'].append(calcular_bss(prob_fold, y_test_fold))
    metricas_cal['ECE'].append(calcular_ece(prob_fold, y_test_fold))
    metricas_cal['LogLoss'].append(log_loss(y_test_fold, prob_fold))

medias_cal = {k: np.mean(v) for k, v in metricas_cal.items()}
std_cal    = {k: np.std(v)  for k, v in metricas_cal.items()}

# 2. Definir la ruta de tu CSV original
RUTA_CSV_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'TABLAS', 'resultados_metricas_multiventana.csv')

# 3. Construir la fila con el formato exacto
# Le ponemos de nombre algo como "XGB_Platt" para distinguirlo del XGB normal
fila_calibrada = {
    'Ventana': 'Medio_6_24',
    'Modelo':  f'XGB_Calibrado', 
    'n_vars':  len(VARIABLES),
}

for metrica in ['AUC_ROC', 'AUC_PR', 'Brier', 'BSS', 'ECE', 'LogLoss']:
    fila_calibrada[metrica]          = round(medias_cal[metrica], 4)
    fila_calibrada[f'{metrica}_std'] = round(std_cal[metrica], 4)

# 4. Añadir la fila al CSV sin sobreescribir lo que ya tienes
pd.DataFrame([fila_calibrada]).to_csv(
    RUTA_CSV_RESULTADOS, mode='a', header=False, index=False
)

print(f"Fila 'XGB_Calibrado' añadida al archivo: {RUTA_CSV_RESULTADOS}")