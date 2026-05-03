import warnings
warnings.filterwarnings('ignore')
 
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
 
RUTA = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
 
df = pd.read_csv(RUTA)
df = df.dropna(subset=['pf_max'])
 
y = df['etiqueta_norad_6_24'].astype(int).values
 
 
def cohens_d(grupo_1, grupo_2):
    n_1, n_2 = len(grupo_1), len(grupo_2)
    s_pooled = np.sqrt(
        ((n_1 - 1) * grupo_1.var(ddof=1) + (n_2 - 1) * grupo_2.var(ddof=1))
        / (n_1 + n_2 - 2)
    )
    return (grupo_1.mean() - grupo_2.mean()) / s_pooled
 
 
def evaluar_estadisticos(variable_base):
    print(f"\n{'=' * 60}")
    print(f"VARIABLE: {variable_base}")
    print(f"{'=' * 60}")
    print(f"{'Estadístico':<20} {'AUC univariante':<18} {'Cohen d':<12} {'Pos vs Neg (mediana)'}")
    print("-" * 60)
 
    resultados = {}
 
    for sufijo in ['_media', '_min', '_max']:
        columna = variable_base + sufijo
        if columna not in df.columns:
            continue
 
        idx_validos = df[columna].notna()
        if idx_validos.sum() < 100:
            continue
 
        y_validos = y[idx_validos]
        x_validos = df[columna][idx_validos].values
 
        auc = roc_auc_score(y_validos, x_validos)
        if auc < 0.5:
            auc = 1 - auc
 
        valores_pos = df.loc[idx_validos & (df['etiqueta_norad_6_24'] == 1), columna]
        valores_neg = df.loc[idx_validos & (df['etiqueta_norad_6_24'] == 0), columna]
        d = cohens_d(valores_pos, valores_neg)
 
        med_pos = valores_pos.median()
        med_neg = valores_neg.median()
        comparacion = '↑' if med_pos > med_neg else ('↓' if med_pos < med_neg else '=')
 
        resultados[sufijo] = (auc, d)
        print(f"{columna:<20} {auc:.4f}             {d:+.3f}        "
              f"{comparacion} (pos={med_pos:.1f}, neg={med_neg:.1f})")
 
    if resultados:
        ganadora = max(resultados.items(), key=lambda kv: kv[1][0])
        print(f"\n  → Mayor AUC univariante: {variable_base}{ganadora[0]} "
              f"(AUC={ganadora[1][0]:.4f})")
 
 
print("ANÁLISIS DE DESEMPATE — min / media / max")
 
evaluar_estadisticos('hr')
evaluar_estadisticos('temp')
evaluar_estadisticos('tp')
evaluar_estadisticos('fio2')
evaluar_estadisticos('bilirrubina')
evaluar_estadisticos('gpt')
evaluar_estadisticos('leucocitos')
evaluar_estadisticos('sofa')
evaluar_estadisticos('glucemia')
evaluar_estadisticos('plaquetas')