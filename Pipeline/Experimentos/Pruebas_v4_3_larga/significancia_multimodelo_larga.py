import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests


RUTA_CSV = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'

CARPETA_BASE   = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
CARPETA_FIGURAS = os.path.join(CARPETA_BASE, 'figuras')
os.makedirs(CARPETA_TABLAS, exist_ok=True)
os.makedirs(CARPETA_FIGURAS, exist_ok=True)

N_REPETICIONES_PERMUTACION = 50   # permutaciones por variable y fold
SEMILLA                    = 42
N_FOLDS_EXTERNO            = 5
N_FOLDS_INTERNO            = 3

VARIABLES_PREDICTORAS = [
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
    'ventilacion_invasiva_12h',
    'gcs_min',

    # Renal (2)
    'creatinina_max',
    'diuresis_ml_kg_12h',

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

ETIQUETA   = 'etiqueta_norad_12_48'
COLUMNA_ID = 'subject_id'

VARIABLES_BINARIAS   = ['gender', 'ventilacion_invasiva_12h']
VARIABLES_CONTINUAS  = [v for v in VARIABLES_PREDICTORAS
                        if v not in VARIABLES_BINARIAS]


def cargar_datos():
    df = pd.read_csv(RUTA_CSV)
    return df


def preparar(df):
    predictores = df[VARIABLES_PREDICTORAS].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)
    etiqueta    = df[ETIQUETA].copy()
    paciente_id = df[COLUMNA_ID].copy()
    return predictores, etiqueta, paciente_id


def calcular_significancia_univariante(df):
    
    df_primera = (
        df.sort_values([COLUMNA_ID, 'contador_estancia_uci'])
          .drop_duplicates(COLUMNA_ID, keep='first')
          .reset_index(drop=True)
    )

    df_test = df_primera.copy()
    df_test['gender'] = (df_test['gender'] == 'M').astype(int)

    n_pos = int((df_test[ETIQUETA] == 1).sum())
    n_neg = int((df_test[ETIQUETA] == 0).sum())

    print(f"  Univariante — primera estancia por paciente: "
          f"{len(df_test)} obs | {n_pos} positivos | {n_neg} negativos")

    filas = []

    for var in VARIABLES_CONTINUAS:
        grupo_pos = df_test.loc[df_test[ETIQUETA] == 1, var]
        grupo_neg = df_test.loc[df_test[ETIQUETA] == 0, var]

        if len(grupo_pos) < 2 or len(grupo_neg) < 2:
            filas.append({
                'variable': var, 'test': 'Mann-Whitney U',
                'n_pos': len(grupo_pos), 'n_neg': len(grupo_neg),
                'estadistico_uni': np.nan, 'p_valor_uni': np.nan,
            })
            continue

        estadistico, p_val = mannwhitneyu(
            grupo_pos, grupo_neg, alternative='two-sided'
        )
        filas.append({
            'variable': var,
            'test': 'Mann-Whitney U',
            'n_pos': len(grupo_pos),
            'n_neg': len(grupo_neg),
            'estadistico_uni': round(estadistico, 2),
            'p_valor_uni': p_val,
        })

    for var in VARIABLES_BINARIAS:
        tabla_contingencia = pd.crosstab(df_test[var], df_test[ETIQUETA])
        chi2, p_val, _, _ = chi2_contingency(tabla_contingencia)
        filas.append({
            'variable': var,
            'test': 'Chi-cuadrado',
            'n_pos': n_pos,
            'n_neg': n_neg,
            'estadistico_uni': round(chi2, 2),
            'p_valor_uni': p_val,
        })

    df_uni = pd.DataFrame(filas)

    mascara_validos = df_uni['p_valor_uni'].notna()
    p_validos = df_uni.loc[mascara_validos, 'p_valor_uni'].values
    rechaza, p_corr, _, _ = multipletests(p_validos, method='fdr_bh', alpha=0.05)

    df_uni['p_valor_uni_BH'] = np.nan
    df_uni['sig_uni_BH'] = False
    df_uni.loc[mascara_validos, 'p_valor_uni_BH'] = p_corr
    df_uni.loc[mascara_validos, 'sig_uni_BH']     = rechaza

    df_uni = df_uni.sort_values('p_valor_uni').reset_index(drop=True)

    n_sig = df_uni['sig_uni_BH'].sum()
    print(f"  Significativas (BH-FDR q<0.05): {n_sig} / {len(df_uni)}")

    return df_uni


def definir_modelos():
    """
    Devuelve lista de dicts con las claves:
      clave, nombre_legible, pipeline, espacio_hiperparametros, n_jobs_grid
    """
    modelos = [

        {
            'clave': 'LR',
            'nombre_legible': 'Regresión Logística',
            'pipeline': Pipeline([
                ('escalador', RobustScaler()),
                ('modelo', LogisticRegression(
                    max_iter=5000,
                    class_weight='balanced',
                    solver='liblinear',
                    random_state=SEMILLA,
                )),
            ]),
            'espacio_hiperparametros': [
                {'modelo__C': [1e-4, 5e-4, 1e-3, 5e-3,
                               1e-2, 5e-2, 0.1, 0.5, 1, 5, 10, 50]}
            ],
            'n_jobs_grid': -1,
        },

        {
            'clave': 'RF',
            'nombre_legible': 'Random Forest',
            'pipeline': Pipeline([
                ('modelo', RandomForestClassifier(
                    random_state=SEMILLA,
                    n_jobs=-1,
                )),
            ]),
            'espacio_hiperparametros': {
                'modelo__n_estimators':     [300, 500, 1000],
                'modelo__max_depth':        [10, 20, None],
                'modelo__min_samples_leaf': [1, 5],
                'modelo__max_features':     ['sqrt'],
                'modelo__class_weight':     ['balanced', 'balanced_subsample'],
            },
            'n_jobs_grid': -1,
        },

        {
            'clave': 'XGB',
            'nombre_legible': 'XGBoost',
            'pipeline': Pipeline([
                ('modelo', XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    random_state=SEMILLA,
                    n_jobs=1,
                    tree_method='hist',
                    verbosity=0,
                )),
            ]),
            'espacio_hiperparametros': {
                'modelo__n_estimators':      [300, 600],
                'modelo__max_depth':         [3, 5, 7],
                'modelo__learning_rate':     [0.01, 0.05, 0.1],
                'modelo__subsample':         [0.8, 1.0],
                'modelo__colsample_bytree':  [0.8, 1.0],
                'modelo__reg_lambda':        [1, 10],
                'modelo__scale_pos_weight':  [1, 5, 9],
            },
            'n_jobs_grid': -1,
        },

        {
            'clave': 'LGBM',
            'nombre_legible': 'LightGBM',
            'pipeline': Pipeline([
                ('modelo', LGBMClassifier(
                    random_state=SEMILLA,
                    verbosity=-1,
                    n_jobs=1,
                    objective='binary',
                )),
            ]),
            'espacio_hiperparametros': {
                'modelo__n_estimators':       [300, 600],
                'modelo__num_leaves':         [31, 63],
                'modelo__learning_rate':      [0.01, 0.05, 0.1],
                'modelo__min_child_samples':  [10, 30],
                'modelo__reg_lambda':         [1, 10],
                'modelo__subsample':          [0.8, 1.0],
                'modelo__class_weight':       ['balanced', None],
            },
            'n_jobs_grid': -1,
        },

        {
            'clave': 'CAT',
            'nombre_legible': 'CatBoost',
            'pipeline': Pipeline([
                ('modelo', CatBoostClassifier(
                    loss_function='Logloss',
                    eval_metric='AUC',
                    random_seed=SEMILLA,
                    verbose=0,
                    thread_count=-1,
                )),
            ]),
            'espacio_hiperparametros': {
                'modelo__iterations':         [500, 1000],
                'modelo__depth':              [5, 6, 7],
                'modelo__learning_rate':      [0.03, 0.05, 0.1],
                'modelo__l2_leaf_reg':        [1, 5, 15],
                'modelo__bagging_temperature':[0, 0.5, 1],
            },
            'n_jobs_grid': 1,   
        },

        {
            'clave': 'NB',
            'nombre_legible': 'Naive Bayes',
            'pipeline': Pipeline([
                ('escalador', RobustScaler()),
                ('modelo', GaussianNB()),
            ]),
            'espacio_hiperparametros': {
                'modelo__var_smoothing': np.logspace(-12, -2, 30),
            },
            'n_jobs_grid': -1,
        },
    ]
    return modelos

def cv_anidada_con_permutation_importance(
        nombre_clave, nombre_legible, pipeline, espacio,
        predictores, etiqueta, paciente_id, n_jobs_grid):
    """
    Ejecuta la CV anidada completa y calcula permutation importance en
    cada fold externo sobre el conjunto de test (datos no vistos).

    Devuelve un dict con:
      - aucs_por_fold        : lista de 5 AUCs
      - auc_medio / auc_desv : estadísticos globales
      - importancias_medias_por_fold : array [5, n_vars]
      - todas_las_permutaciones_por_fold : lista de 5 arrays [n_vars, n_reps]
      - nombres_variables    : lista de strings
    """
    cv_externo = StratifiedGroupKFold(
        n_splits=N_FOLDS_EXTERNO, shuffle=True, random_state=SEMILLA)
    cv_interno = StratifiedGroupKFold(
        n_splits=N_FOLDS_INTERNO, shuffle=True, random_state=SEMILLA)

    nombres_variables = predictores.columns.tolist()
    n_vars = len(nombres_variables)

    aucs_por_fold                   = []
    importancias_medias_por_fold    = np.zeros((N_FOLDS_EXTERNO, n_vars))
    todas_las_permutaciones_por_fold = []

    tiempo_inicio = time.time()

    for num_fold, (indices_train, indices_test) in enumerate(
            cv_externo.split(predictores, etiqueta, groups=paciente_id), start=1):

        x_train       = predictores.iloc[indices_train]
        x_test        = predictores.iloc[indices_test]
        y_train       = etiqueta.iloc[indices_train]
        y_test        = etiqueta.iloc[indices_test]
        pacientes_train = paciente_id.iloc[indices_train]

        busqueda = GridSearchCV(
            estimator=pipeline,
            param_grid=espacio,
            cv=cv_interno,
            scoring='roc_auc',
            n_jobs=n_jobs_grid,
            refit=True,
        )
        busqueda.fit(x_train, y_train, groups=pacientes_train)

        probabilidades = busqueda.predict_proba(x_test)[:, 1]
        auc_fold       = roc_auc_score(y_test, probabilidades)
        aucs_por_fold.append(auc_fold)

        print(f"    Fold {num_fold}: AUC={auc_fold:.4f} | "
              f"best_params: {busqueda.best_params_}")

        # Permutation importance sobre test (datos no vistos por el modelo)
        resultado_perm = permutation_importance(
            busqueda.best_estimator_,
            x_test, y_test,
            scoring='roc_auc',
            n_repeats=N_REPETICIONES_PERMUTACION,
            random_state=SEMILLA,
            n_jobs=-1,
        )

        importancias_medias_por_fold[num_fold - 1, :] = (
            resultado_perm.importances_mean
        )
        # importances tiene shape [n_vars, n_reps]
        todas_las_permutaciones_por_fold.append(resultado_perm.importances)

    tiempo_min = (time.time() - tiempo_inicio) / 60
    auc_medio = np.mean(aucs_por_fold)
    auc_desv  = np.std(aucs_por_fold)

    print(f"\n  {nombre_legible} — AUC medio: {auc_medio:.4f} ± {auc_desv:.4f}"
          f"  (tiempo: {tiempo_min:.1f} min)\n")

    return {
        'clave':                          nombre_clave,
        'nombre_legible':                 nombre_legible,
        'aucs_por_fold':                  aucs_por_fold,
        'auc_medio':                      auc_medio,
        'auc_desv':                       auc_desv,
        'importancias_medias_por_fold':   importancias_medias_por_fold,
        'todas_las_permutaciones_por_fold': todas_las_permutaciones_por_fold,
        'nombres_variables':              nombres_variables,
    }


def construir_tabla_un_modelo(resultados_modelo):
    """
    A partir del dict devuelto por cv_anidada_con_permutation_importance,
    genera un DataFrame con una fila por variable.
    """
    nombres      = resultados_modelo['nombres_variables']
    imp_folds    = resultados_modelo['importancias_medias_por_fold']
    todas_perm   = resultados_modelo['todas_las_permutaciones_por_fold']
    clave_modelo = resultados_modelo['clave']
    auc_modelo   = resultados_modelo['auc_medio']

    filas = []
    for i, nombre in enumerate(nombres):
        importancias_5_folds = imp_folds[:, i]
        media = importancias_5_folds.mean()
        desv  = importancias_5_folds.std()

        # IC95% como percentiles de los 5 valores de fold
        ic_inf = np.percentile(importancias_5_folds, 2.5)
        ic_sup = np.percentile(importancias_5_folds, 97.5)

        # P-valor empírico: fracción de permutaciones (todos los folds) con
        # caída de AUC <= 0. Variable sin efecto real → p ≈ 0.5.
        permutaciones_concatenadas = np.concatenate(
            [m[i, :] for m in todas_perm]
        )
        p_emp = np.mean(permutaciones_concatenadas <= 0)

        filas.append({
            'modelo':           clave_modelo,
            'auc_modelo':       auc_modelo,
            'variable':         nombre,
            'caida_AUC_media':  round(media, 6),
            'caida_AUC_pp':     round(media * 100, 4),
            'caida_AUC_desv':   round(desv, 6),
            'caida_AUC_IC_inf': round(ic_inf, 6),
            'caida_AUC_IC_sup': round(ic_sup, 6),
            'p_valor_empirico': round(p_emp, 4),
            'IC_excluye_cero':  bool(ic_inf > 0),
            'p_emp_menor_005':  bool(p_emp < 0.05),
        })

    return pd.DataFrame(filas).sort_values(
        'caida_AUC_media', ascending=False
    ).reset_index(drop=True)


def construir_tabla_consenso(df_todos_modelos, claves_modelos):
    
    variables = (df_todos_modelos['variable']
                 .drop_duplicates()
                 .tolist())

    filas_consenso = []
    for var in variables:
        fila = {'variable': var}
        n_significativa = 0
        for clave in claves_modelos:
            sub = df_todos_modelos[
                (df_todos_modelos['variable'] == var) &
                (df_todos_modelos['modelo']   == clave)
            ]
            if len(sub) == 0:
                fila[f'{clave}_caida_pp']   = np.nan
                fila[f'{clave}_sig']        = False
            else:
                fila[f'{clave}_caida_pp'] = sub.iloc[0]['caida_AUC_pp']
                fila[f'{clave}_sig']       = sub.iloc[0]['IC_excluye_cero']
                if sub.iloc[0]['IC_excluye_cero']:
                    n_significativa += 1

        fila['n_modelos_significativos'] = n_significativa
        n_total = len(claves_modelos)
        if n_significativa == n_total:
            fila['consenso'] = 'Unanime_significativa'
        elif n_significativa >= 4:
            fila['consenso'] = 'Mayoria_significativa'
        elif n_significativa > 0:
            fila['consenso'] = 'Discordante'
        else:
            fila['consenso'] = 'Unanime_no_significativa'

        filas_consenso.append(fila)

    df_consenso = pd.DataFrame(filas_consenso)
    df_consenso = df_consenso.sort_values(
        'n_modelos_significativos', ascending=False
    ).reset_index(drop=True)
    return df_consenso


def construir_tabla_resumen_global(df_consenso, df_univariante, claves_modelos):
    
    # Seleccionar sólo lo necesario del univariante
    df_uni_slim = df_univariante[
        ['variable', 'test', 'p_valor_uni', 'p_valor_uni_BH', 'sig_uni_BH']
    ].copy()

    # Fusionar
    df_resumen = df_uni_slim.merge(df_consenso, on='variable', how='outer')

    # Consenso global: considera tanto univariante como multimodelo
    def categorizar_global(fila):
        uni  = bool(fila['sig_uni_BH'])
        n_ml = int(fila['n_modelos_significativos']) \
               if pd.notna(fila['n_modelos_significativos']) else 0
        n_total = len(claves_modelos)

        if uni and n_ml == n_total:
            return 'Robusta_completa'          # uni + todos los modelos
        elif uni and n_ml >= 4:
            return 'Robusta_mayoria'           # uni + mayoría de modelos
        elif uni and n_ml > 0:
            return 'Moderada'                  # uni + algún modelo
        elif not uni and n_ml >= 4:
            return 'ML_mayoria_sin_uni'        # no uni pero mayoría ML
        elif uni and n_ml == 0:
            return 'Solo_univariante'          # sólo univariante
        elif not uni and n_ml > 0:
            return 'Solo_ML'                   # sólo algún modelo ML
        else:
            return 'No_significativa'          # ningún análisis

    df_resumen['consenso_global'] = df_resumen.apply(categorizar_global, axis=1)

    # Orden: por n_modelos_sig desc, luego p univariante asc
    df_resumen = df_resumen.sort_values(
        ['n_modelos_significativos', 'p_valor_uni'],
        ascending=[False, True]
    ).reset_index(drop=True)

    return df_resumen


def graficar_significancia_global(df_resumen_global, df_todos_modelos,
                                  claves_modelos, nombres_legibles_por_clave,
                                  ruta_figura):

    tabla_caida = df_todos_modelos.pivot_table(
        index='variable', columns='modelo', values='caida_AUC_pp'
    )
    tabla_sig_ml = df_todos_modelos.pivot_table(
        index='variable', columns='modelo', values='IC_excluye_cero'
    )

    # Orden de variables: igual que en la tabla resumen global
    orden_variables = df_resumen_global['variable'].tolist()
    # Filtrar a variables presentes en el pivote
    orden_variables = [v for v in orden_variables if v in tabla_caida.index]

    tabla_caida   = tabla_caida.loc[orden_variables, claves_modelos]
    tabla_sig_ml  = tabla_sig_ml.loc[orden_variables, claves_modelos]

    # Univariante alineado al mismo orden
    uni_dict = df_resumen_global.set_index('variable')['sig_uni_BH'].to_dict()
    uni_valores = np.array(
        [1.0 if uni_dict.get(v, False) else 0.0 for v in orden_variables]
    ).reshape(-1, 1)

    # Renombrar columnas del panel derecho
    columnas_legibles = [nombres_legibles_por_clave[c] for c in claves_modelos]
    tabla_caida.columns  = columnas_legibles
    tabla_sig_ml.columns = columnas_legibles

    n_vars   = len(orden_variables)
    n_modelos = len(claves_modelos)

    ancho_uni   = 1.2
    ancho_ml    = n_modelos * 1.5
    fig, (ax_uni, ax_ml) = plt.subplots(
        1, 2,
        figsize=(ancho_uni + ancho_ml + 1.5, max(10, n_vars * 0.45)),
        gridspec_kw={'width_ratios': [ancho_uni, ancho_ml]},
    )

    cmap_binario = matplotlib.colors.ListedColormap(['#d62728', '#2ca02c'])
    sns.heatmap(
        uni_valores,
        cmap=cmap_binario,
        vmin=0, vmax=1,
        annot=False,
        linewidths=0.5,
        linecolor='gray',
        cbar=False,
        ax=ax_uni,
        yticklabels=orden_variables,
        xticklabels=['Univariante'],
    )
    # Añadir ★ / · en cada celda
    for i, var in enumerate(orden_variables):
        sig = uni_dict.get(var, False)
        ax_uni.text(
            0.5, i + 0.5,
            '★' if sig else '·',
            ha='center', va='center',
            color='white', fontsize=10, fontweight='bold',
        )
    ax_uni.set_ylabel('')
    ax_uni.set_title('Uni\n(BH)', fontsize=9)
    ax_uni.tick_params(axis='y', labelsize=8)
    ax_uni.tick_params(axis='x', labelsize=8, rotation=0)


    vmax_abs = max(abs(tabla_caida.values.max()),
                   abs(tabla_caida.values.min()), 0.5)
    sns.heatmap(
        tabla_caida,
        cmap='RdYlGn',
        center=0,
        vmin=-vmax_abs,
        vmax=vmax_abs,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7},
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={
            'label': 'Caída AUC (puntos porcentuales)',
            'shrink': 0.6,
        },
        yticklabels=False,   # ya aparecen en el panel izquierdo
        ax=ax_ml,
    )
    # ★ en celdas donde IC95% excluye cero
    for i, var in enumerate(orden_variables):
        for j, col in enumerate(columnas_legibles):
            if tabla_sig_ml.loc[var, col]:
                ax_ml.text(
                    j + 0.85, i + 0.22, '★',
                    ha='center', va='center',
                    color='black', fontsize=8, fontweight='bold',
                )
    ax_ml.set_ylabel('')
    ax_ml.set_title(
        f'Permutation Importance — 6 modelos\n'
        f'★ = IC95% excluye 0 | '
        f'{N_FOLDS_EXTERNO} folds externos | '
        f'{N_REPETICIONES_PERMUTACION} permutaciones/fold',
        fontsize=9,
    )
    ax_ml.tick_params(axis='x', labelsize=8, rotation=30)

    fig.suptitle(
        'Tabla de significancia global — set reducido v4 (26 variables)\n'
        'Univariante (Mann-Whitney / Chi²) + Permutation Importance multimodelo',
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figura guardada en: {ruta_figura}")


def main():
    print("=" * 70)
    print("SIGNIFICANCIA GLOBAL — UNIVARIANTE + PERMUTATION IMPORTANCE MULTIMODELO")
    print(f"Set reducido v4 ({len(VARIABLES_PREDICTORAS)} variables)")
    print("=" * 70)
    print(f"  [1] Univariante  : Mann-Whitney U / Chi² + FDR BH")
    print(f"  [2] Multimodelo  : LR, RF, XGB, LGBM, CAT, NB")
    print(f"  CV externa       : {N_FOLDS_EXTERNO} folds (StratifiedGroupKFold)")
    print(f"  CV interna       : {N_FOLDS_INTERNO} folds (búsqueda hiperparámetros)")
    print(f"  Permutaciones    : {N_REPETICIONES_PERMUTACION} por variable y fold")
    print(f"  Criterio ML      : IC95% de la caída de AUC excluye 0")
    print()

    df = cargar_datos()
    predictores, etiqueta, paciente_id = preparar(df)

    print(f"Dataset completo: {predictores.shape} | "
          f"Positivos: {etiqueta.sum()} ({100 * etiqueta.mean():.2f}%)")
    print(f"Pacientes únicos: {paciente_id.nunique()} | "
          f"Estancias: {len(predictores)}")
    print()

    tiempo_global = time.time()

    print("─" * 70)
    print("[1/2] ANÁLISIS UNIVARIANTE")
    print("─" * 70)
    df_univariante = calcular_significancia_univariante(df)

    print()
    print("  Resultados univariantes (ordenados por p-valor):")
    print(f"  {'Variable':<28} {'Test':<18} {'p_valor_uni':>12} "
          f"{'p_BH':>10} {'Sig':>5}")
    print("  " + "─" * 75)
    for _, fila in df_univariante.iterrows():
        p_str  = f"{fila['p_valor_uni']:.2e}" \
                 if fila['p_valor_uni'] < 1e-3 else f"{fila['p_valor_uni']:.4f}"
        pb_str = f"{fila['p_valor_uni_BH']:.2e}" \
                 if pd.notna(fila['p_valor_uni_BH']) \
                    and fila['p_valor_uni_BH'] < 1e-3 \
                 else (f"{fila['p_valor_uni_BH']:.4f}"
                       if pd.notna(fila['p_valor_uni_BH']) else 'NaN')
        sig_str = '★' if fila['sig_uni_BH'] else '·'
        print(f"  {fila['variable']:<28} {fila['test']:<18} "
              f"{p_str:>12} {pb_str:>10} {sig_str:>5}")
    print()

    print("─" * 70)
    print("[2/2] PERMUTATION IMPORTANCE MULTIMODELO")
    print("─" * 70)
    print()

    modelos = definir_modelos()
    claves_modelos             = [m['clave'] for m in modelos]
    nombres_legibles_por_clave = {m['clave']: m['nombre_legible'] for m in modelos}

    lista_tablas    = []
    aucs_por_modelo = {}

    for especificacion in modelos:
        clave       = especificacion['clave']
        nombre_leg  = especificacion['nombre_legible']
        pipeline    = especificacion['pipeline']
        espacio     = especificacion['espacio_hiperparametros']
        n_jobs_grid = especificacion['n_jobs_grid']

        print("─" * 60)
        print(f"MODELO: {nombre_leg}  ({clave})")
        print("─" * 60)

        resultados_modelo = cv_anidada_con_permutation_importance(
            clave, nombre_leg, pipeline, espacio,
            predictores, etiqueta, paciente_id, n_jobs_grid,
        )

        aucs_por_modelo[clave] = (
            resultados_modelo['auc_medio'],
            resultados_modelo['auc_desv'],
        )

        tabla_modelo = construir_tabla_un_modelo(resultados_modelo)
        lista_tablas.append(tabla_modelo)

        n_sig = tabla_modelo['IC_excluye_cero'].sum()
        print(f"  Variables con IC95% excluye 0: {n_sig} / {len(tabla_modelo)}")
        for _, fila in tabla_modelo[tabla_modelo['IC_excluye_cero']].iterrows():
            print(f"    ★ {fila['variable']:<28} caída = {fila['caida_AUC_pp']:.3f} pp")
        print()

    df_todos_modelos = pd.concat(lista_tablas, ignore_index=True)
    df_consenso      = construir_tabla_consenso(df_todos_modelos, claves_modelos)

    df_resumen_global = construir_tabla_resumen_global(
        df_consenso, df_univariante, claves_modelos
    )

    tiempo_horas = (time.time() - tiempo_global) / 3600
    print()
    print("=" * 70)
    print(f"TABLA RESUMEN GLOBAL (tiempo total: {tiempo_horas:.2f} h)")
    print("=" * 70)
    print()

    print("AUC por modelo:")
    for clave in claves_modelos:
        auc_m, auc_s = aucs_por_modelo[clave]
        print(f"  {clave:<6} {nombres_legibles_por_clave[clave]:<25} "
              f"AUC = {auc_m:.4f} ± {auc_s:.4f}")
    print()

    # Cabecera de la tabla resumen
    cabecera  = f"  {'Variable':<28} {'Uni':>4}"
    for clave in claves_modelos:
        cabecera += f" {clave:>5}"
    cabecera += f"  {'n_ML':>5}  Consenso_global"
    separador = "  " + "─" * (len(cabecera) - 2)
    print(cabecera)
    print(separador)

    for _, fila in df_resumen_global.iterrows():
        uni_s = '★' if fila['sig_uni_BH'] else '·'
        linea = f"  {fila['variable']:<28} {uni_s:>4}"
        for clave in claves_modelos:
            col_sig = f'{clave}_sig'
            simbolo = '★' if (col_sig in fila.index and fila[col_sig]) else '·'
            linea += f" {simbolo:>5}"
        n_ml = int(fila['n_modelos_significativos']) \
               if pd.notna(fila['n_modelos_significativos']) else 0
        linea += f"  {n_ml:>5}  {fila['consenso_global']}"
        print(linea)

    print()
    print(f"  Leyenda: ★ = significativa | · = no significativa")
    print(f"  Uni = Mann-Whitney U / Chi² con FDR BH (primera estancia/paciente)")
    print(f"  ML  = IC95% de permutation importance excluye 0 (todas las estancias)")
    print()

    # Resumen por categoría de consenso global
    print("Variables por categoría de consenso global:")
    orden_cats = [
        'Robusta_completa', 'Robusta_mayoria', 'Moderada',
        'ML_mayoria_sin_uni', 'Solo_univariante', 'Solo_ML',
        'No_significativa',
    ]
    descripcion_cats = {
        'Robusta_completa':    'Uni ★ + TODOS los modelos ML ★ (máxima solidez)',
        'Robusta_mayoria':     'Uni ★ + mayoría de modelos ML ★ (≥4/6)',
        'Moderada':            'Uni ★ + algún modelo ML ★',
        'ML_mayoria_sin_uni':  'No sig. univariante, pero mayoría ML ★ (efecto ajustado)',
        'Solo_univariante':    'Sig. univariante, ningún modelo ML ★',
        'Solo_ML':             'No sig. univariante, algún modelo ML ★',
        'No_significativa':    'No significativa en ningún análisis',
    }
    for cat in orden_cats:
        vars_cat = df_resumen_global[
            df_resumen_global['consenso_global'] == cat
        ]['variable'].tolist()
        if not vars_cat:
            continue
        print(f"\n  [{descripcion_cats[cat]}] → {len(vars_cat)} variable(s):")
        for v in vars_cat:
            print(f"    - {v}")
    print()

    ruta_uni = os.path.join(
        CARPETA_TABLAS, 'significancia_univariante_multimodelo_v4.csv'
    )
    df_univariante.to_csv(ruta_uni, index=False)
    print(f"Tabla univariante guardada en        : {ruta_uni}")

    ruta_tabla_modelos = os.path.join(
        CARPETA_TABLAS, 'permutation_importance_multimodelo_v4.csv'
    )
    df_todos_modelos.to_csv(ruta_tabla_modelos, index=False)
    print(f"Tabla por modelo guardada en         : {ruta_tabla_modelos}")

    ruta_resumen_global = os.path.join(
        CARPETA_TABLAS, 'tabla_resumen_global_significancia_v4.csv'
    )
    df_resumen_global.to_csv(ruta_resumen_global, index=False)
    print(f"Tabla resumen global guardada en     : {ruta_resumen_global}")

    ruta_figura = os.path.join(
        CARPETA_FIGURAS, 'significancia_global_v4.png'
    )
    graficar_significancia_global(
        df_resumen_global, df_todos_modelos,
        claves_modelos, nombres_legibles_por_clave,
        ruta_figura,
    )

    print()
    print(f"Tiempo total: {tiempo_horas:.2f} h")
    print("=" * 70)


if __name__ == "__main__":
    main()