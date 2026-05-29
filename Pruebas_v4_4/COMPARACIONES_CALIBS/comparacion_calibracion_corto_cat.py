import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')


RUTA_CSV    = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv'
RUTA_PKL    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MODELOS_ENTRENADOS\modelo_Corto_3_12_CAT.pkl')
ETIQUETA    = 'etiqueta_norad_3_12'
COLUMNA_ID  = 'subject_id'
VARIABLES   = ['map_min', 'pf_min', 'sofa_max', 'tp_max']
N_SPLITS    = 5
RANDOM_SEED = 42
N_BINS_CAL  = 10   # bins para ECE y curva de calibración

def calcular_ece(probabilidades, etiquetas, n_bins=N_BINS_CAL):
    """Expected Calibration Error con bins de igual anchura."""
    limites = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(etiquetas)
    for i in range(n_bins):
        mascara = (probabilidades >= limites[i]) & (probabilidades < limites[i + 1])
        n_bin = mascara.sum()
        if n_bin == 0:
            continue
        confianza = probabilidades[mascara].mean()
        precision = etiquetas[mascara].mean()
        ece += (n_bin / n_total) * abs(confianza - precision)
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

print("Cargando datos y modelo...")
df      = pd.read_csv(RUTA_CSV)
y       = df[ETIQUETA].values.astype(int)
grupos  = df[COLUMNA_ID].values
X       = df[VARIABLES].copy()
modelo  = joblib.load(RUTA_PKL)

print(f"  Pacientes: {len(y)} | Positivos: {y.sum()} ({y.mean():.3f})")

# Mismo StratifiedGroupKFold que en graficos_oof.py
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

prob_sin_cal  = np.zeros(len(y))
prob_platt    = np.zeros(len(y))
prob_isoton   = np.zeros(len(y))

print(f"\nGenerando probabilidades OOF con {N_SPLITS} folds...")

from sklearn.base import clone

for fold_idx, (idx_train, idx_test) in enumerate(cv.split(X, y, grupos)):
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    modelo_fold = clone(modelo)
    modelo_fold.fit(X_train, y_train)

    prob_bruta_train = modelo_fold.predict_proba(X_train)[:, 1]
    prob_bruta_test  = modelo_fold.predict_proba(X_test)[:, 1]
    prob_sin_cal[idx_test] = prob_bruta_test

    #Calibración de Platt: ajuste en train del fold, aplicación en test
    calibrador_platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrador_platt.fit(prob_bruta_train.reshape(-1, 1), y_train)
    prob_platt[idx_test] = calibrador_platt.predict_proba(
        prob_bruta_test.reshape(-1, 1)
    )[:, 1]

    #Calibración isotónica: ajuste en train del fold, aplicación en test
    calibrador_iso = IsotonicRegression(out_of_bounds='clip')
    calibrador_iso.fit(prob_bruta_train, y_train)
    prob_isoton[idx_test] = calibrador_iso.predict(prob_bruta_test)

    print(f"  Fold {fold_idx + 1}/{N_SPLITS} completado "
          f"(test n={len(idx_test)}, positivos={y_test.sum()})")

print("\n=== MÉTRICAS OOF COMPARATIVAS ===")
resultados = [
    metricas_resumen(prob_sin_cal, y, 'Sin calibrar'),
    metricas_resumen(prob_platt,   y, 'Platt'),
    metricas_resumen(prob_isoton,  y, 'Isotónica'),
]
df_resultados = pd.DataFrame(resultados)
print()
print(df_resultados.to_string(index=False))

fig = plt.figure(figsize=(16, 5))
fig.suptitle(
    'Comparación empírica calibración OOF — CAT | Corto_3_12',
    fontsize=14, fontweight='bold'
)
gs = gridspec.GridSpec(1, 3, figure=fig)

colores = {'Sin calibrar': '#ff7f0e', 'Platt': '#1f77b4', 'Isotónica': '#2ca02c'}
probs_dict = {
    'Sin calibrar': prob_sin_cal,
    'Platt':        prob_platt,
    'Isotónica':    prob_isoton,
}

ax1 = fig.add_subplot(gs[0])
for nombre, probs in probs_dict.items():
    fraccion_pos, media_pred = calibration_curve(
        y, probs, n_bins=N_BINS_CAL, strategy='quantile'
    )
    ax1.plot(media_pred, fraccion_pos, marker='o', lw=2,
             color=colores[nombre], label=nombre, markersize=6)
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfecta')
# Limitar eje X al rango real de probabilidades para no mostrar espacio vacío
prob_max = max(prob_sin_cal.max(), prob_platt.max(), prob_isoton.max())
ax1.set_xlim(-0.01, min(prob_max + 0.05, 1.0))
ax1.set_ylim(-0.01, 1.0)
ax1.set_xlabel('Probabilidad predicha media')
ax1.set_ylabel('Fracción de positivos')
ax1.set_title('Curvas de calibración (bins por cuantil)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.4)

# Panel 2: ECE comparativo (barras)
ax2 = fig.add_subplot(gs[1])
nombres = [r['nombre'] for r in resultados]
eces    = [r['ece']    for r in resultados]
barras  = ax2.bar(nombres, eces,
                  color=[colores[n] for n in nombres], alpha=0.85, edgecolor='black')
ax2.axhline(y=0.03, color='red', lw=1.5, linestyle='--', label='ECE=0.03 (umbral)')
ax2.set_ylabel('ECE')
ax2.set_title('Expected Calibration Error\n(menor es mejor)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.4, axis='y')
for barra, valor in zip(barras, eces):
    ax2.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.001,
             f'{valor:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 3: BSS comparativo (barras)
ax3 = fig.add_subplot(gs[2])
bss_vals = [r['bss'] for r in resultados]
barras3  = ax3.bar(nombres, bss_vals,
                   color=[colores[n] for n in nombres], alpha=0.85, edgecolor='black')
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
                           'calibracion_comparativa_corto_cat.png')
plt.savefig(ruta_figura, dpi=200, bbox_inches='tight')
plt.show()
print(f"\nGráfica guardada en: {ruta_figura}")

ece_base   = df_resultados.loc[df_resultados['nombre'] == 'Sin calibrar', 'ece'].values[0]
ece_mejor  = df_resultados['ece'].min()
mejor_cal  = df_resultados.loc[df_resultados['ece'].idxmin(), 'nombre']
mejora_ece = ece_base - ece_mejor
bss_base   = df_resultados.loc[df_resultados['nombre'] == 'Sin calibrar', 'bss'].values[0]
bss_mejor  = df_resultados.loc[df_resultados['ece'].idxmin(), 'bss'].values[0]

if mejor_cal == 'Sin calibrar':
    print("  La calibración NO mejora el modelo. Usar sin calibrar.")
elif mejora_ece < 0.005:
    print(f"  Mejora de ECE con {mejor_cal}: {mejora_ece:.4f} (marginal, <0.005).")
    print("  La calibración apenas aporta")
else:
    perdida_bss = bss_base - bss_mejor
    print(f"  Mejor calibración: {mejor_cal} | Mejora ECE: {mejora_ece:.4f}")
    if perdida_bss > 0.005:
        print(f" calibrar mejora ECE pero empeora BSS en {perdida_bss:.4f}.")
        print("  Evaluar tradeoff")
    else:
        print(f"  Calibrar con {mejor_cal} mejora ECE sin penalizar BSS.")
