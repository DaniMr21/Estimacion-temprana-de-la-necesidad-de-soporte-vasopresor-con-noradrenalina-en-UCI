import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


# CONFIGURACIÓN

RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'

CARPETA_BASE = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

ETIQUETA = 'etiqueta_norad_6_24'

variables_predictoras = [
    # Demografía y contexto (4)
    'anchor_age',
    'gender',
    'peso_kg',
    'contador_estancia_uci',

    # Hemodinámica (2)
    'map_min',
    'hr_media',

    # Respiratorio (4)
    'pf_min',
    'spo2_min',
    'fio2_max',
    'rr_max',

    # Ventilación y conciencia (2)
    'ventilacion_invasiva_6h',
    'gcs_min',

    # Renal (2)
    'creatinina_max',
    'diuresis_ml_kg_6h',

    # Ácido-base (3)
    'lactato_max',
    'ph_min',
    'bicarbonato_min',

    # Hepático (2)
    'bilirrubina_media',
    'gpt_max',

    # Coagulación (2)
    'tp_max',
    'plaquetas_min',

    # Hematología/inflamación (2)
    'leucocitos_min',
    'hemoglobina_min',

    # Metabólico (1)
    'glucemia_min',

    # Otro vital (1)
    'temp_min',

    # Gravedad global (1)
    'sofa_max',
]

# 1. CARGA Y FILTRADO A PRIMERA ESTANCIA POR PACIENTE

print("--------------------------------------------------")
print("REGRESIÓN LOGÍSTICA MULTIVARIANTE — SET REDUCIDO v4 (26 variables)")
print("--------------------------------------------------")

df = pd.read_csv(RUTA_CSV)
df = df.dropna(subset=['pf_max'])
print(f"\nDataset cargado:")
print(f"  Estancias totales      : {len(df)}")
print(f"  Pacientes únicos       : {df['subject_id'].nunique()}")

df = (df.sort_values(['subject_id', 'contador_estancia_uci'])
        .drop_duplicates('subject_id', keep='first')
        .reset_index(drop=True))
print(f"  Tras filtrar 1ª estancia por paciente:")
print(f"    Estancias            : {len(df)}")
print(f"    Positivos (etiq=1)   : {df[ETIQUETA].sum()} "
      f"({100*df[ETIQUETA].mean():.2f}%)")
print()

# 2. PREPARACIÓN

X = df[variables_predictoras].copy()
X['gender'] = (X['gender'] == 'M').astype(int)
y = df[ETIQUETA].astype(int)

# Estandarización: coeficientes comparables entre variables
escalador = StandardScaler()
X_std = pd.DataFrame(
    escalador.fit_transform(X),
    columns=X.columns,
    index=X.index,
)

# 3. AJUSTE DE LA REGRESIÓN LOGÍSTICA

print("--------------------------------------------------")
print("[1/3] Ajustando regresión logística multivariante")
print("--------------------------------------------------")

X_sm = sm.add_constant(X_std)
modelo = sm.Logit(y, X_sm).fit(disp=0)

print(f"  Pseudo R²    : {modelo.prsquared:.4f}")
print(f"  Log-Likelihood: {modelo.llf:.2f}")
print(f"  LLR p-value  : {modelo.llr_pvalue:.2e}")
print(f"  AIC          : {modelo.aic:.2f}")
print(f"  BIC          : {modelo.bic:.2f}")
print(f"  Convergencia : {'OK' if modelo.mle_retvals['converged'] else 'NO CONVERGIÓ'}")

# 4. TABLA DE RESULTADOS

print("--------------------------------------------------")
print("[2/3] Construyendo tabla de coeficientes")
print("--------------------------------------------------")

ic = modelo.conf_int()
ic.columns = ['IC_inf', 'IC_sup']

tabla = pd.DataFrame({
    'variable': modelo.params.index,
    'coef': modelo.params.values,
    'std_err': modelo.bse.values,
    'IC_inf_coef': ic['IC_inf'].values,
    'IC_sup_coef': ic['IC_sup'].values,
    'OR': np.exp(modelo.params.values),
    'OR_IC_inf': np.exp(ic['IC_inf'].values),
    'OR_IC_sup': np.exp(ic['IC_sup'].values),
    'p_valor': modelo.pvalues.values,
})

# Quitar la constante para los contrastes y la corrección
tabla_vars = tabla[tabla['variable'] != 'const'].copy().reset_index(drop=True)

# Corrección por comparaciones múltiples
rechaza, p_corr, _, _ = multipletests(tabla_vars['p_valor'], method='fdr_bh', alpha=0.05)
tabla_vars['p_valor_BH'] = p_corr
tabla_vars['significativa_bruta'] = tabla_vars['p_valor'] < 0.05
tabla_vars['significativa_BH'] = rechaza

# Ordenar por p-valor
tabla_vars = tabla_vars.sort_values('p_valor').reset_index(drop=True)

# Formateo para impresión
def formatea_p(p):
    if pd.isna(p):
        return 'NaN'
    if p < 1e-4:
        return f'{p:.2e}'
    return f'{p:.4f}'

tabla_print = tabla_vars[['variable', 'coef', 'OR', 'OR_IC_inf', 'OR_IC_sup',
                          'p_valor', 'p_valor_BH', 'significativa_BH']].copy()
tabla_print['coef'] = tabla_print['coef'].round(3)
tabla_print['OR'] = tabla_print['OR'].round(3)
tabla_print['OR_IC_inf'] = tabla_print['OR_IC_inf'].round(3)
tabla_print['OR_IC_sup'] = tabla_print['OR_IC_sup'].round(3)
tabla_print['p_valor'] = tabla_print['p_valor'].apply(formatea_p)
tabla_print['p_valor_BH'] = tabla_print['p_valor_BH'].apply(formatea_p)

print()
print("Tabla de coeficientes (ordenada por p-valor):")
print()
print(tabla_print.to_string(index=False))
print()

# 5. RESUMEN

n_total = len(tabla_vars)
n_sig_brutas = tabla_vars['significativa_bruta'].sum()
n_sig_bh = tabla_vars['significativa_BH'].sum()

print("RESUMEN MULTIVARIANTE")

print(f"  Variables analizadas              : {n_total}")
print(f"  Significativas (p < 0.05 sin corregir): {n_sig_brutas}")
print(f"  Significativas (BH-FDR q < 0.05)      : {n_sig_bh}")
print()

print(f"  Significativas tras BH-FDR (multivariante):")
sig_multi = tabla_vars[tabla_vars['significativa_BH']]['variable'].tolist()
for v in sig_multi:
    fila = tabla_vars[tabla_vars['variable'] == v].iloc[0]
    direccion = "↑ riesgo" if fila['coef'] > 0 else "↓ riesgo"
    print(f"    - {v:<28} OR={fila['OR']:.2f}  [{fila['OR_IC_inf']:.2f}, "
          f"{fila['OR_IC_sup']:.2f}]  ({direccion})")
print()

print(f"  NO significativas tras BH-FDR (multivariante):")
no_sig_multi = tabla_vars[~tabla_vars['significativa_BH']]['variable'].tolist()
for v in no_sig_multi:
    print(f"    - {v}")
print()


# 6. COMPARACIÓN UNIVARIANTE vs MULTIVARIANTE

print("[3/3] Comparación con análisis univariante")

ruta_univariante = os.path.join(CARPETA_TABLAS, 'significancia_univariante_v4_reducido.csv')

if os.path.exists(ruta_univariante):
    uni = pd.read_csv(ruta_univariante)[['variable', 'p_valor', 'p_valor_BH',
                                          'significativa_BH']]
    uni.columns = ['variable', 'p_uni', 'p_uni_BH', 'sig_uni_BH']

    multi = tabla_vars[['variable', 'p_valor', 'p_valor_BH',
                        'significativa_BH', 'coef', 'OR']].copy()
    multi.columns = ['variable', 'p_multi', 'p_multi_BH', 'sig_multi_BH',
                     'coef_multi', 'OR_multi']

    comparacion = uni.merge(multi, on='variable', how='outer')

    # Categorías de comparación
    def categoriza(row):
        if row['sig_uni_BH'] and row['sig_multi_BH']:
            return 'Sig. en ambos (robusta)'
        elif row['sig_uni_BH'] and not row['sig_multi_BH']:
            return 'Solo univariante (absorbida)'
        elif not row['sig_uni_BH'] and row['sig_multi_BH']:
            return 'Solo multivariante (efecto ajustado)'
        else:
            return 'No sig. en ninguno'

    comparacion['categoria'] = comparacion.apply(categoriza, axis=1)
    comparacion = comparacion.sort_values('p_multi').reset_index(drop=True)

    print()
    print("Comparación de significancia univariante vs multivariante:")
    print()
    for cat in ['Sig. en ambos (robusta)',
                'Solo univariante (absorbida)',
                'Solo multivariante (efecto ajustado)',
                'No sig. en ninguno']:
        variables_cat = comparacion[comparacion['categoria'] == cat]['variable'].tolist()
        print(f"  [{cat}] ({len(variables_cat)}):")
        for v in variables_cat:
            print(f"    - {v}")
        print()

    ruta_comp = os.path.join(CARPETA_TABLAS,
                             'comparacion_univariante_vs_multivariante_v4.csv')
    comparacion.to_csv(ruta_comp, index=False)
    print(f"  Comparación guardada en: {ruta_comp}")
else:
    print(f"  No se encontró el fichero univariante en: {ruta_univariante}")
    print(f"  Ejecuta primero significancia_univariante_v4.py para tener la comparación.")

print()

# 7. GUARDADO

ruta_salida = os.path.join(CARPETA_TABLAS,
                           'regresion_logistica_multivariante_v4_reducido.csv')
tabla_vars.to_csv(ruta_salida, index=False)
print(f"Tabla multivariante guardada en: {ruta_salida}")