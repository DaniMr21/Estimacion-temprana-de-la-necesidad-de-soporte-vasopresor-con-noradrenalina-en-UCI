import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
RUTA_CSV   = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'
RUTA_PKL   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'MODELOS_ENTRENADOS', 'modelo_Largo_12_48_XGB.pkl')
ETIQUETA   = 'etiqueta_norad_12_48'
COLUMNA_ID = 'subject_id'
VARIABLES  = ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min',
              'map_min', 'glucemia_min', 'sofa_max']
N_SPLITS    = 5
RANDOM_SEED = 42
N_BINS_CAL  = 10

# ── FUNCIONES MÉTRICAS ─────────────────────────────────────────────────────────

def calcular_ece(probabilidades, etiquetas, n_bins=N_BINS_CAL):
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


def metricas_resumen(probabilidades, etiquetas, nombre):
    auc = roc_auc_score(etiquetas, probabilidades)
    bs  = brier_score_loss(etiquetas, probabilidades)
    bss = calcular_bss(probabilidades, etiquetas)
    ece = calcular_ece(probabilidades, etiquetas)
    print(f"  [{nombre:12s}]  AUC={auc:.4f}  Brier={bs:.4f}  BSS={bss:+.4f}  ECE={ece:.4f}")
    return {'nombre': nombre, 'auc': auc, 'brier': bs, 'bss': bss, 'ece': ece}

# ── CARGA ──────────────────────────────────────────────────────────────────────
print("Cargando datos y modelo...")
df     = pd.read_csv(RUTA_CSV).dropna(subset=['pf_max'])
y      = df[ETIQUETA].values.astype(int)
grupos = df[COLUMNA_ID].values
X      = df[VARIABLES].copy()
modelo = joblib.load(RUTA_PKL)

print(f"  Pacientes: {len(y)} | Positivos: {y.sum()} ({y.mean():.3f})")

# ── PROBABILIDADES OOF HONESTAS ────────────────────────────────────────────────
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

prob_sin_cal = np.zeros(len(y))
prob_platt   = np.zeros(len(y))
prob_isoton  = np.zeros(len(y))

print(f"\nGenerando probabilidades OOF con {N_SPLITS} folds...")

for fold_idx, (idx_train, idx_test) in enumerate(cv.split(X, y, grupos)):
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    modelo_fold = clone(modelo)
    modelo_fold.fit(X_train, y_train)

    prob_bruta_train           = modelo_fold.predict_proba(X_train)[:, 1]
    prob_bruta_test            = modelo_fold.predict_proba(X_test)[:, 1]
    prob_sin_cal[idx_test]     = prob_bruta_test

    cal_platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    cal_platt.fit(prob_bruta_train.reshape(-1, 1), y_train)
    prob_platt[idx_test] = cal_platt.predict_proba(prob_bruta_test.reshape(-1, 1))[:, 1]

    cal_iso = IsotonicRegression(out_of_bounds='clip')
    cal_iso.fit(prob_bruta_train, y_train)
    prob_isoton[idx_test] = cal_iso.predict(prob_bruta_test)

    print(f"  Fold {fold_idx + 1}/{N_SPLITS} — test n={len(idx_test)}, positivos={y_test.sum()}")

# ── MÉTRICAS ───────────────────────────────────────────────────────────────────
print("\n=== MÉTRICAS OOF COMPARATIVAS ===")
resultados = [
    metricas_resumen(prob_sin_cal, y, 'Sin calibrar'),
    metricas_resumen(prob_platt,   y, 'Platt'),
    metricas_resumen(prob_isoton,  y, 'Isotónica'),
]
df_resultados = pd.DataFrame(resultados)

# ── GRÁFICA ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))
fig.suptitle('Comparación empírica calibración OOF — XGB | Largo_12_48',
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(1, 3, figure=fig)

colores    = {'Sin calibrar': '#ff7f0e', 'Platt': '#1f77b4', 'Isotónica': '#2ca02c'}
probs_dict = {'Sin calibrar': prob_sin_cal, 'Platt': prob_platt, 'Isotónica': prob_isoton}

ax1 = fig.add_subplot(gs[0])
for nombre, probs in probs_dict.items():
    frac_pos, media_pred = calibration_curve(y, probs, n_bins=N_BINS_CAL, strategy='quantile')
    ax1.plot(media_pred, frac_pos, marker='o', lw=2, color=colores[nombre],
             label=nombre, markersize=6)
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfecta')
prob_max = max(p.max() for p in probs_dict.values())
ax1.set_xlim(-0.01, min(prob_max + 0.05, 1.0))
ax1.set_ylim(-0.01, 1.0)
ax1.set_xlabel('Probabilidad predicha media')
ax1.set_ylabel('Fracción de positivos')
ax1.set_title('Curvas de calibración (bins por cuantil)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.4)

ax2 = fig.add_subplot(gs[1])
nombres = [r['nombre'] for r in resultados]
eces    = [r['ece']    for r in resultados]
barras  = ax2.bar(nombres, eces, color=[colores[n] for n in nombres],
                  alpha=0.85, edgecolor='black')
ax2.axhline(y=0.03, color='red', lw=1.5, linestyle='--', label='ECE=0.03 (umbral)')
ax2.set_ylabel('ECE')
ax2.set_title('Expected Calibration Error\n(menor es mejor)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.4, axis='y')
for barra, valor in zip(barras, eces):
    ax2.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.001,
             f'{valor:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3 = fig.add_subplot(gs[2])
bss_vals = [r['bss'] for r in resultados]
barras3  = ax3.bar(nombres, bss_vals, color=[colores[n] for n in nombres],
                   alpha=0.85, edgecolor='black')
ax3.axhline(y=0, color='red', lw=1.5, linestyle='--', label='BSS=0 (modelo nulo)')
ax3.set_ylabel('BSS')
ax3.set_title('Brier Skill Score\n(mayor es mejor)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.4, axis='y')
for barra, valor in zip(barras3, bss_vals):
    offset = 0.001 if valor >= 0 else -0.003
    ax3.text(barra.get_x() + barra.get_width() / 2, valor + offset,
             f'{valor:+.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
ruta_figura = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'calibracion_comparativa_largo_xgb.png')
plt.savefig(ruta_figura, dpi=200, bbox_inches='tight')
plt.show()
print(f"\nGráfica guardada en: {ruta_figura}")

# ── VEREDICTO ──────────────────────────────────────────────────────────────────
ece_base  = df_resultados.loc[df_resultados['nombre'] == 'Sin calibrar', 'ece'].values[0]
ece_mejor = df_resultados['ece'].min()
mejor_cal = df_resultados.loc[df_resultados['ece'].idxmin(), 'nombre']
mejora    = ece_base - ece_mejor
bss_base  = df_resultados.loc[df_resultados['nombre'] == 'Sin calibrar', 'bss'].values[0]
bss_mejor = df_resultados.loc[df_resultados['ece'].idxmin(), 'bss'].values[0]

print("\n=== VEREDICTO ===")
if mejor_cal == 'Sin calibrar':
    print("  La calibración NO mejora el modelo. Usar sin calibrar.")
elif mejora < 0.005:
    print(f"  Mejora ECE con {mejor_cal}: {mejora:.4f} (marginal).")
    print("  La calibración apenas aporta. Recomendado: usar sin calibrar.")
else:
    perdida_bss = bss_base - bss_mejor
    print(f"  Mejor calibrador: {mejor_cal} | Mejora ECE: {mejora:.4f}")
    if perdida_bss > 0.005:
        print(f"  ADVERTENCIA: mejora ECE pero empeora BSS en {perdida_bss:.4f}. Evaluar tradeoff.")
    else:
        print(f"  Calibrar con {mejor_cal} mejora ECE sin penalizar BSS. Recomendado.")
