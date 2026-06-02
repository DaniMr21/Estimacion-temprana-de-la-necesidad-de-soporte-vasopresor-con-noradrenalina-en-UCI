import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# CONFIGURACIÓN

CARPETA_BASE = os.path.dirname(__file__) if '__file__' in dir() else '.'
CARPETA_TABLAS = os.path.join(CARPETA_BASE, 'tablas')
CARPETA_FIGURAS = os.path.join(CARPETA_BASE, 'figuras')
os.makedirs(CARPETA_TABLAS, exist_ok=True)
os.makedirs(CARPETA_FIGURAS, exist_ok=True)

N_REPETICIONES_PERMUTACION = 50  # nº permutaciones por variable y fold
RANDOM_STATE = 42


# CARGA Y PREPARACIÓN (idéntico a baseline_v4_2.py)

def cargar_datos():
    ruta = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv'
    df = pd.read_csv(ruta)
    df = df.dropna(subset=['pf_max'])
    return df


def preparar(df):

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

    predictores = df[variables_predictoras].copy()
    etiqueta = df['etiqueta_norad_6_24'].copy()
    paciente_id = df['subject_id'].copy()
    predictores['gender'] = (predictores['gender'] == 'M').astype(int)

    return predictores, etiqueta, paciente_id


# CV ANIDADO + PERMUTATION IMPORTANCE

def cv_anidada_con_permutation_importance(predictores, etiqueta, paciente_id):

    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                      random_state=RANDOM_STATE)
    cv_interno = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                      random_state=RANDOM_STATE)

    pipeline_random_forest = Pipeline([
        ('modelo', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    espacio_random_forest = {
        'modelo__n_estimators': [300, 400, 500, 600, 700, 850, 1000],
        'modelo__max_depth': [None, 5, 10, 20, 30],
        'modelo__min_samples_leaf': [1, 2, 5],
        'modelo__max_features': ['sqrt', 0.3, 0.5],
        'modelo__class_weight': ['balanced', 'balanced_subsample'],
    }

    nombres_variables = predictores.columns.tolist()
    n_vars = len(nombres_variables)

    aucs_por_fold = []
    mejores_params_por_fold = []
    # Matriz [n_folds, n_vars] con la importancia media de cada fold
    importancias_medias_por_fold = np.zeros((5, n_vars))
    # Matriz [n_folds, n_vars] con desv estándar de las permutaciones de ese fold
    importancias_std_por_fold = np.zeros((5, n_vars))
    # Lista de matrices [n_repeticiones, n_vars] por fold (para p-valor empírico)
    todas_las_permutaciones_por_fold = []

    tiempo_inicio = time.time()

    for num_fold, (indices_train, indices_test) in enumerate(
            cv_externo.split(predictores, etiqueta, groups=paciente_id), start=1):

        x_train = predictores.iloc[indices_train]
        x_test = predictores.iloc[indices_test]
        y_train = etiqueta.iloc[indices_train]
        y_test = etiqueta.iloc[indices_test]
        pacientes_train = paciente_id.iloc[indices_train]

        # Búsqueda de hiperparámetros igual que baseline_v4_2.py
        busqueda = GridSearchCV(
            estimator=pipeline_random_forest,
            param_grid=espacio_random_forest,
            cv=cv_interno,
            scoring='roc_auc',
            n_jobs=-1,
            refit=True,
        )
        busqueda.fit(x_train, y_train, groups=pacientes_train)

        probabilidades = busqueda.predict_proba(x_test)[:, 1]
        auc_fold = roc_auc_score(y_test, probabilidades)
        aucs_por_fold.append(auc_fold)
        mejores_params_por_fold.append(busqueda.best_params_)

        print(f"  Fold {num_fold}: AUC={auc_fold:.4f}")
        print(f"    Best params: {busqueda.best_params_}")

        # Permutation importance sobre el conjunto de test (datos no vistos)
        print(f"    Calculando permutation importance "
              f"({N_REPETICIONES_PERMUTACION} permutaciones por variable)...")

        resultado_perm = permutation_importance(
            busqueda.best_estimator_,
            x_test, y_test,
            scoring='roc_auc',
            n_repeats=N_REPETICIONES_PERMUTACION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        importancias_medias_por_fold[num_fold - 1, :] = resultado_perm.importances_mean
        importancias_std_por_fold[num_fold - 1, :] = resultado_perm.importances_std
        # Guardar la matriz completa para el p-valor empírico
        todas_las_permutaciones_por_fold.append(resultado_perm.importances)

        # Top-5 variables del fold actual
        orden = np.argsort(-resultado_perm.importances_mean)
        print(f"    Top-5 variables del fold:")
        for i in orden[:5]:
            print(f"      {nombres_variables[i]:<28} "
                  f"caída AUC = {resultado_perm.importances_mean[i]:.4f}")
        print()

    tiempo_min = (time.time() - tiempo_inicio) / 60
    auc_medio = np.mean(aucs_por_fold)
    auc_desv = np.std(aucs_por_fold)

    print(f"Random Forest — AUC medio: {auc_medio:.4f} ± {auc_desv:.4f}  "
          f"(tiempo: {tiempo_min:.1f} min)")

    return {
        'aucs_por_fold': aucs_por_fold,
        'auc_medio': auc_medio,
        'auc_desv': auc_desv,
        'mejores_params_por_fold': mejores_params_por_fold,
        'importancias_medias_por_fold': importancias_medias_por_fold,
        'importancias_std_por_fold': importancias_std_por_fold,
        'todas_las_permutaciones_por_fold': todas_las_permutaciones_por_fold,
        'nombres_variables': nombres_variables,
    }


# AGREGACIÓN Y P-VALOR EMPÍRICO

def construir_tabla_resumen(resultados):

    nombres = resultados['nombres_variables']
    importancias_folds = resultados['importancias_medias_por_fold']
    todas_perm = resultados['todas_las_permutaciones_por_fold']

    n_vars = len(nombres)
    filas = []

    for i, nombre in enumerate(nombres):
        # Importancias medias de los 5 folds
        importancias_5_folds = importancias_folds[:, i]
        media = importancias_5_folds.mean()
        desv = importancias_5_folds.std()

        # IC95% bootstrap a partir de los 5 valores de fold
        ic_inf = np.percentile(importancias_5_folds, 2.5) \
            if len(importancias_5_folds) >= 5 else np.nan
        ic_sup = np.percentile(importancias_5_folds, 97.5) \
            if len(importancias_5_folds) >= 5 else np.nan

        # P-valor empírico: fracción de permutaciones (sumando todos los folds)
        # con caída de AUC ≤ 0. Si la variable no aporta nada, esperaríamos
        # que la mitad de las permutaciones den caída ≤ 0 (p ≈ 0.5).
        permutaciones_concatenadas = np.concatenate(
            [m[i, :] for m in todas_perm]
        )
        p_emp = np.mean(permutaciones_concatenadas <= 0)

        # Caída media en pp de AUC
        media_pp = media * 100

        filas.append({
            'variable': nombre,
            'caida_AUC_media': media,
            'caida_AUC_pp': media_pp,
            'caida_AUC_desv': desv,
            'caida_AUC_IC_inf': ic_inf,
            'caida_AUC_IC_sup': ic_sup,
            'p_valor_empirico': p_emp,
            'IC_excluye_cero': (ic_inf > 0),
            'p_emp_menor_005': (p_emp < 0.05),
        })

    df_resumen = pd.DataFrame(filas)
    df_resumen = df_resumen.sort_values('caida_AUC_media', ascending=False) \
                           .reset_index(drop=True)
    return df_resumen


# VISUALIZACIÓN

def graficar_importancias(df_resumen, ruta_figura, auc_medio):

    df_plot = df_resumen.sort_values('caida_AUC_media').copy()

    fig, ax = plt.subplots(figsize=(10, 9))

    colores = ['#2ca02c' if row['IC_excluye_cero'] else '#d62728'
               for _, row in df_plot.iterrows()]

    posiciones = np.arange(len(df_plot))
    ax.barh(
        posiciones,
        df_plot['caida_AUC_pp'],
        xerr=df_plot['caida_AUC_desv'] * 100,
        color=colores,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.85,
        capsize=3,
    )
    ax.set_yticks(posiciones)
    ax.set_yticklabels(df_plot['variable'])
    ax.set_xlabel('Caída de AUC (puntos porcentuales) al permutar la variable')
    ax.set_title(
        f'Permutation Importance — Random Forest (set reducido v4, 26 variables)\n'
        f'AUC base = {auc_medio:.4f} | Promedio sobre 5 folds del CV externo | '
        f'{N_REPETICIONES_PERMUTACION} permutaciones por fold'
    )
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Leyenda manual
    from matplotlib.patches import Patch
    leyenda = [
        Patch(facecolor='#2ca02c', edgecolor='black',
              label='IC95% excluye 0 (importante)'),
        Patch(facecolor='#d62728', edgecolor='black',
              label='IC95% incluye 0 (no aporta)'),
    ]
    ax.legend(handles=leyenda, loc='lower right')

    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=150, bbox_inches='tight')
    plt.close()


# MAIN

def main():
    print("--------------")
    print("PERMUTATION IMPORTANCE — RANDOM FOREST (set reducido v4)")
    print("--------------")

    df = cargar_datos()
    predictores, etiqueta, paciente_id = preparar(df)

    print(f"Dataset: {predictores.shape} | "
          f"Positivos: {etiqueta.sum()} ({100*etiqueta.mean():.2f}%)")
    print(f"Pacientes únicos: {paciente_id.nunique()} | "
          f"Estancias: {len(predictores)}")
    print(f"Variables predictoras: {predictores.shape[1]}")
    print(f"Permutaciones por variable y fold: {N_REPETICIONES_PERMUTACION}")
    print()

    tiempo_global = time.time()

    print("--- Random Forest (con permutation importance) ---")
    resultados = cv_anidada_con_permutation_importance(
        predictores, etiqueta, paciente_id
    )

    # Tabla resumen
    print()
    print("--------------")
    print("TABLA DE PERMUTATION IMPORTANCE (agregada sobre 5 folds)")
    print("--------------")
    df_resumen = construir_tabla_resumen(resultados)

    df_print = df_resumen.copy()
    df_print['caida_AUC_pp'] = df_print['caida_AUC_pp'].round(3)
    df_print['caida_AUC_desv'] = df_print['caida_AUC_desv'].round(4)
    df_print['caida_AUC_IC_inf'] = df_print['caida_AUC_IC_inf'].round(4)
    df_print['caida_AUC_IC_sup'] = df_print['caida_AUC_IC_sup'].round(4)
    df_print['p_valor_empirico'] = df_print['p_valor_empirico'].round(4)

    columnas_print = ['variable', 'caida_AUC_pp', 'caida_AUC_desv',
                      'caida_AUC_IC_inf', 'caida_AUC_IC_sup',
                      'p_valor_empirico', 'IC_excluye_cero', 'p_emp_menor_005']
    print(df_print[columnas_print].to_string(index=False))
    print()

    # Resumen
    n_importantes = df_resumen['IC_excluye_cero'].sum()
    n_p_sig = df_resumen['p_emp_menor_005'].sum()

    print("--------------")
    print("RESUMEN")
    print("--------------")
    print(f"  AUC del modelo (5 folds)              : "
          f"{resultados['auc_medio']:.4f} ± {resultados['auc_desv']:.4f}")
    print(f"  Variables con IC95% que excluye el 0  : {n_importantes} / "
          f"{len(df_resumen)}")
    print(f"  Variables con p empírico < 0.05       : {n_p_sig} / "
          f"{len(df_resumen)}")
    print()

    print("Variables IMPORTANTES (IC95% excluye 0):")
    for _, fila in df_resumen[df_resumen['IC_excluye_cero']].iterrows():
        print(f"  - {fila['variable']:<28} "
              f"caída AUC = {fila['caida_AUC_pp']:.2f} pp")
    print()

    print("Variables NO IMPORTANTES (IC95% incluye 0):")
    for _, fila in df_resumen[~df_resumen['IC_excluye_cero']].iterrows():
        print(f"  - {fila['variable']:<28} "
              f"caída AUC = {fila['caida_AUC_pp']:.2f} pp")
    print()

    # Guardado
    ruta_tabla = os.path.join(CARPETA_TABLAS,
                              'permutation_importance_RF_v4_reducido.csv')
    df_resumen.to_csv(ruta_tabla, index=False)
    print(f"Tabla guardada en  : {ruta_tabla}")

    ruta_figura = os.path.join(CARPETA_FIGURAS,
                               'permutation_importance_RF_v4_reducido.png')
    graficar_importancias(df_resumen, ruta_figura, resultados['auc_medio'])
    print(f"Figura guardada en : {ruta_figura}")

    tiempo_total_horas = (time.time() - tiempo_global) / 3600
    print()
    print(f"Tiempo total: {tiempo_total_horas:.2f} h")


if __name__ == "__main__":
    main()
