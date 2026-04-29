import warnings
warnings.filterwarnings('ignore')
 
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
 
# CARGA

RUTA = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
 
df = pd.read_csv(RUTA)
df = df.dropna(subset=['pf_max'])
 
y = df['etiqueta_norad_6_24'].astype(int).values
 
# FUNCIONES AUXILIARES

def cohens_d(grupo_1, grupo_2):
    """Cohen's d para dos muestras independientes con varianzas comparables.
 
    d = (media_1 - media_2) / desviación tipica conjunta
    """
    n_1, n_2 = len(grupo_1), len(grupo_2)
    s_pooled = np.sqrt(
        ((n_1 - 1) * grupo_1.var(ddof=1) + (n_2 - 1) * grupo_2.var(ddof=1))
        / (n_1 + n_2 - 2)
    )
    return (grupo_1.mean() - grupo_2.mean()) / s_pooled
 
 
def evaluar_estadisticos(variable_base, sentido_clinico):
    """Para una variable base (p. ej. 'sofa'), recorre los sufijos
    _media / _min / _max, calcula AUC univariante y Cohen's d entre
    clases, y reporta el ganador."""
 
    print(f"\n{'=' * 70}")
    print(f"VARIABLE: {variable_base}")
    print(f"Sentido clínico esperado: {sentido_clinico}")
    print(f"{'=' * 70}")
 
    print(f"{'Estadístico':<20} {'AUC univariante':<18} "
          f"{'Cohen d':<12} {'Pos vs Neg (mediana)':<22}")
    print("-" * 70)
 
    resultados = {}
 
    for sufijo in ['_media', '_min', '_max']:
        columna = variable_base + sufijo
        if columna not in df.columns:
            continue
 
        # Filtrar a estancias con valor no nulo en esta variable
        idx_validos = df[columna].notna()
        if idx_validos.sum() < 100:
            continue
 
        y_validos = y[idx_validos]
        x_validos = df[columna][idx_validos].values
 
        # AUC univariante (con autocorrección de dirección)
        auc = roc_auc_score(y_validos, x_validos)
        if auc < 0.5:
            auc = 1 - auc  # mantenemos siempre la dirección informativa
 
        # Cohen's d con signo
        valores_pos = df.loc[idx_validos & (df['etiqueta_norad_6_24'] == 1), columna]
        valores_neg = df.loc[idx_validos & (df['etiqueta_norad_6_24'] == 0), columna]
        d = cohens_d(valores_pos, valores_neg)
 
        # Medianas por clase
        med_pos = valores_pos.median()
        med_neg = valores_neg.median()
        if med_pos > med_neg:
            comparacion = '↑'
        elif med_pos < med_neg:
            comparacion = '↓'
        else:
            comparacion = '='
 
        resultados[sufijo] = (auc, d)
        print(f"{variable_base + sufijo:<20} "
              f"{auc:.4f}             "
              f"{d:+.3f}        "
              f"{comparacion} (pos={med_pos:.1f}, neg={med_neg:.1f})")
 
    if resultados:
        ganadora = max(resultados.items(), key=lambda kv: kv[1][0])
        print(f"\n  → Mayor AUC univariante: "
              f"{variable_base}{ganadora[0]} (AUC={ganadora[1][0]:.4f})")
 
# EJECUCIÓN

print("------------------------------")
print("# ANÁLISIS DE VARIABLES DUDOSAS (desempate min/media/max)")

 
# Variables donde la decisión entre estadísticos no es clínicamente
# inequívoca y conviene apoyarse en datos.
evaluar_estadisticos('hr',          'Taquicardia compensatoria → max esperable; '
                                    'pero los picos pueden ser ruido (dolor, agitación)')
evaluar_estadisticos('fio2',        'Mayor soporte respiratorio = peor → max esperable')
evaluar_estadisticos('bilirrubina', 'Disfunción hepática → max o media (poca variación en 6h)')
evaluar_estadisticos('gpt',         'Daño hepatocelular agudo → pico (max) esperable')
evaluar_estadisticos('tp',          'Coagulopatía → TP alargado (max)')
evaluar_estadisticos('leucocitos',  'Inflamación → max; sepsis grave → min (leucopenia)')
evaluar_estadisticos('sofa',        'Gravedad → max captura el peor momento')
evaluar_estadisticos('glucemia',    'Hiperglucemia de estrés → max; '
                                    'hipoglucemia grave → min')
evaluar_estadisticos('temp',        'Fiebre → max; hipotermia (shock avanzado) → min')
evaluar_estadisticos('plaquetas',   'CID/consumo → min (la caída es la señal)')