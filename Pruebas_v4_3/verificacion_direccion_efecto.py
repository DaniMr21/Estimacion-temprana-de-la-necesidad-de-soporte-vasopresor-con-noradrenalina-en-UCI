import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
os.makedirs(CARPETA_TABLAS, exist_ok=True)

ETIQUETA   = 'etiqueta_norad_6_24'
COLUMNA_ID = 'subject_id'

# Todas las variables del set reducido (para la LR multivariante ajustada)
VARIABLES_MODELO = [
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    'map_min', 'hr_media',
    'diuresis_ml_kg_6h', 'creatinina_max',
    'ph_min', 'bicarbonato_min', 'lactato_max',
    'gpt_max',
    'glucemia_min', 'temp_min',
    'sofa_max',
    'ventilacion_invasiva_6h', 'gender',
]

VARIABLES_REVISAR = VARIABLES_MODELO

VARIABLES_BINARIAS = ['gender', 'ventilacion_invasiva_6h']


def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    # Primera estancia por paciente para evitar dependencia
    df = (df.sort_values([COLUMNA_ID, 'contador_estancia_uci'])
            .drop_duplicates(COLUMNA_ID, keep='first')
            .reset_index(drop=True))
    df['gender'] = (df['gender'] == 'M').astype(int)
    return df


def calcular_or_crudo(df, var, etiqueta):
    """OR crudo de regresión logística univariante."""
    try:
        x = df[[var]].copy()
        escalador = StandardScaler()
        x_std = escalador.fit_transform(x)
        x_sm  = sm.add_constant(x_std)
        modelo = sm.Logit(df[etiqueta], x_sm).fit(disp=0)
        ic = modelo.conf_int()
        return {
            'OR_crudo':     round(float(np.exp(modelo.params[1])), 3),
            'OR_crudo_inf': round(float(np.exp(ic.iloc[1, 0])), 3),
            'OR_crudo_sup': round(float(np.exp(ic.iloc[1, 1])), 3),
            'p_crudo':      float(modelo.pvalues[1]),
        }
    except Exception:
        return {
            'OR_crudo': np.nan, 'OR_crudo_inf': np.nan,
            'OR_crudo_sup': np.nan, 'p_crudo': np.nan,
        }


def calcular_or_ajustado(df, etiqueta):
    """OR ajustado de regresión logística multivariante con todas las variables."""
    x = df[VARIABLES_MODELO].copy()
    escalador = StandardScaler()
    x_std = pd.DataFrame(
        escalador.fit_transform(x),
        columns=x.columns, index=x.index
    )
    x_sm  = sm.add_constant(x_std)
    modelo = sm.Logit(df[etiqueta], x_sm).fit(disp=0)
    ic = modelo.conf_int()

    resultados = {}
    for var in VARIABLES_REVISAR:
        if var in modelo.params.index:
            resultados[var] = {
                'coef_ajustado':  round(float(modelo.params[var]), 3),
                'OR_ajustado':    round(float(np.exp(modelo.params[var])), 3),
                'OR_aj_inf':      round(float(np.exp(ic.loc[var, 0])), 3),
                'OR_aj_sup':      round(float(np.exp(ic.loc[var, 1])), 3),
                'p_ajustado':     float(modelo.pvalues[var]),
            }
        else:
            resultados[var] = {
                'coef_ajustado': np.nan, 'OR_ajustado': np.nan,
                'OR_aj_inf': np.nan, 'OR_aj_sup': np.nan,
                'p_ajustado': np.nan,
            }
    return resultados


def formatea_p(p):
    if pd.isna(p):
        return 'NaN'
    if p < 0.0001:
        return f'{p:.2e}'
    return f'{p:.4f}'


def main():
    print("=" * 70)
    print("VERIFICACIÓN DE DIRECCIÓN DEL EFECTO — DATOS CRUDOS")
    print("=" * 70)

    df = cargar_datos()
    positivos = df[df[ETIQUETA] == 1]
    negativos = df[df[ETIQUETA] == 0]

    print(f"\nDataset: {len(df)} pacientes (primera estancia)")
    print(f"  Positivos : {len(positivos)} ({100*len(positivos)/len(df):.1f}%)")
    print(f"  Negativos : {len(negativos)} ({100*len(negativos)/len(df):.1f}%)")
    print()

    # OR ajustado para todas las variables de una vez
    print("Calculando regresión logística multivariante ajustada...")
    or_ajustado = calcular_or_ajustado(df, ETIQUETA)
    print("Hecho.\n")

    filas = []

    for var in VARIABLES_REVISAR:
        es_binaria = var in VARIABLES_BINARIAS

        # ── Descriptivos ───────────────────────────────────────────────
        if es_binaria:
            prop_pos = positivos[var].mean()
            prop_neg = negativos[var].mean()
            tabla = pd.crosstab(df[var], df[ETIQUETA])
            chi2, p_test, _, _ = chi2_contingency(tabla)
            estadistico = chi2
            nombre_test = 'Chi-cuadrado'
            val_pos_str = f'{prop_pos:.3f} ({prop_pos*100:.1f}%)'
            val_neg_str = f'{prop_neg:.3f} ({prop_neg*100:.1f}%)'
            val_pos = prop_pos
            val_neg = prop_neg
        else:
            val_pos = positivos[var].median()
            val_neg = negativos[var].median()
            p25_pos = positivos[var].quantile(0.25)
            p75_pos = positivos[var].quantile(0.75)
            p25_neg = negativos[var].quantile(0.25)
            p75_neg = negativos[var].quantile(0.75)
            stat, p_test = mannwhitneyu(
                positivos[var],
                negativos[var],
                alternative='two-sided'
            )
            estadistico = stat
            nombre_test = 'Mann-Whitney U'
            val_pos_str = f'{val_pos:.1f} [IQR: {p25_pos:.1f}–{p75_pos:.1f}]'
            val_neg_str = f'{val_neg:.1f} [IQR: {p25_neg:.1f}–{p75_neg:.1f}]'

        # ── OR crudo ───────────────────────────────────────────────────
        or_crudo = calcular_or_crudo(df, var, ETIQUETA)

        # ── OR ajustado ────────────────────────────────────────────────
        or_aj = or_ajustado[var]

        # ── Dirección ─────────────────────────────────────────────────
        dir_datos   = '↑ mayor en positivos' if val_pos >= val_neg \
                      else '↓ menor en positivos'
        dir_or_aj   = '↑ riesgo' if or_aj['coef_ajustado'] > 0 \
                      else '↓ protector aparente'
        coherente   = (val_pos >= val_neg) == (or_aj['coef_ajustado'] > 0)

        # ── Impresión ──────────────────────────────────────────────────
        print(f"{'─' * 60}")
        print(f"VARIABLE: {var}  ({nombre_test})")
        print(f"{'─' * 60}")
        print(f"  Positivos (n={len(positivos[var].dropna())}) : {val_pos_str}")
        print(f"  Negativos (n={len(negativos[var].dropna())}) : {val_neg_str}")
        print(f"  Dirección en datos   : {dir_datos}")
        print(f"  p-valor ({nombre_test}) : {formatea_p(p_test)}")
        print()
        print(f"  OR crudo (univariante, estandarizado):")
        print(f"    OR = {or_crudo['OR_crudo']:.3f}  "
              f"[{or_crudo['OR_crudo_inf']:.3f} – {or_crudo['OR_crudo_sup']:.3f}]  "
              f"p = {formatea_p(or_crudo['p_crudo'])}")
        print()
        print(f"  OR ajustado (LR multivariante, estandarizado):")
        print(f"    Coef = {or_aj['coef_ajustado']:.3f}")
        print(f"    OR   = {or_aj['OR_ajustado']:.3f}  "
              f"[{or_aj['OR_aj_inf']:.3f} – {or_aj['OR_aj_sup']:.3f}]  "
              f"p = {formatea_p(or_aj['p_ajustado'])}")
        print(f"    Dirección ajustada  : {dir_or_aj}")
        print()
        print(f"  Coherencia datos vs OR ajustado : "
              f"{'SÍ' if coherente else 'NO — EFECTO INVERSO'}")
        print()

        filas.append({
            'variable':          var,
            'tipo':              'binaria' if es_binaria else 'continua',
            'test':              nombre_test,
            'n_positivos':       len(positivos[var]),
            'n_negativos':       len(negativos[var]),
            'valor_positivos':   round(val_pos, 4),
            'valor_negativos':   round(val_neg, 4),
            'p_test':            p_test,
            'OR_crudo':          or_crudo['OR_crudo'],
            'OR_crudo_IC_inf':   or_crudo['OR_crudo_inf'],
            'OR_crudo_IC_sup':   or_crudo['OR_crudo_sup'],
            'p_crudo':           or_crudo['p_crudo'],
            'coef_ajustado':     or_aj['coef_ajustado'],
            'OR_ajustado':       or_aj['OR_ajustado'],
            'OR_aj_IC_inf':      or_aj['OR_aj_inf'],
            'OR_aj_IC_sup':      or_aj['OR_aj_sup'],
            'p_ajustado':        or_aj['p_ajustado'],
            'coherente':         coherente,
        })

    # ── Tabla resumen ──────────────────────────────────────────────────
    df_res = pd.DataFrame(filas)

    print("=" * 70)
    print("TABLA RESUMEN")
    print("=" * 70)
    cols_print = ['variable', 'valor_positivos', 'valor_negativos',
                  'p_test', 'OR_crudo', 'OR_ajustado',
                  'OR_aj_IC_inf', 'OR_aj_IC_sup', 'p_ajustado', 'coherente']
    tabla_print = df_res[cols_print].copy()
    tabla_print['p_test']     = tabla_print['p_test'].apply(formatea_p)
    tabla_print['p_ajustado'] = tabla_print['p_ajustado'].apply(formatea_p)
    print(tabla_print.to_string(index=False))
    print()

    # ── Guardado ───────────────────────────────────────────────────────
    ruta_salida = os.path.join(
        CARPETA_TABLAS, 'verificacion_direccion_efecto_v4.csv'
    )
    df_res.to_csv(ruta_salida, index=False)
    print(f"\nTabla guardada en: {ruta_salida}")


if __name__ == "__main__":
    main()
