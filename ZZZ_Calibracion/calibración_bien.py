import warnings
warnings.filterwarnings('ignore')

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

DIR_SCRIPT      = os.path.dirname(os.path.abspath(__file__))
CARPETA_TABLAS  = os.path.join(DIR_SCRIPT, 'evaluacion_final', 'tablas')
CARPETA_FIGURAS = os.path.join(DIR_SCRIPT, 'evaluacion_final', 'figuras')
for carpeta in [CARPETA_TABLAS, CARPETA_FIGURAS]:
    os.makedirs(carpeta, exist_ok=True)

RANDOM_STATE             = 42
N_SPLITS_EXTERNOS        = 5
N_SPLITS_INTERNOS        = 3
PROPORCIONES_CALIBRACION = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
METODOS_CALIBRACION      = ['sigmoid', 'isotonic']
N_BINS_CALIBRACION       = 5
ESTRATEGIA_BINS          = 'quantile'


# ─────────────────────────────────────────────
# VARIABLES
# ─────────────────────────────────────────────

VARIABLES_V4_6_24 = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    'ventilacion_invasiva_6h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_6h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]

VARIABLES_V4P_3_12 = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'fio2_max', 'rr_max',
    'ventilacion_invasiva_3h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_3h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]

VARIABLES_V4L_12_48 = [
    'anchor_age', 'gender', 'peso_kg', 'contador_estancia_uci',
    'map_min', 'hr_media',
    'pf_min', 'spo2_min', 'rr_max',
    'ventilacion_invasiva_12h', 'gcs_min',
    'creatinina_max', 'diuresis_ml_kg_12h',
    'lactato_max', 'ph_min', 'bicarbonato_min',
    'bilirrubina_media', 'gpt_max',
    'tp_max', 'plaquetas_min',
    'leucocitos_min', 'hemoglobina_min',
    'glucemia_min', 'temp_min', 'sofa_max',
]


# ─────────────────────────────────────────────
# VENTANAS
# calibrar=True  → RF: búsqueda proporción × método dentro de cada fold.
# calibrar=False → CatBoost: sin calibración.
# ─────────────────────────────────────────────

VENTANAS = {

    'v4_6_24_rf': {
        'ruta_csv':    r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'variables':   VARIABLES_V4_6_24,
        'etiqueta':    'etiqueta_norad_6_24',
        'calibrar':    True,
        'n_jobs_grid': -1,
        'modelo': RandomForestClassifier(
            # class_weight y max_features fijados por ser ganadores en
            # prácticamente todos los folds del análisis previo.
            class_weight='balanced_subsample',
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'espacio': {
            # 5 × 3 × 3 = 45 combinaciones × 3 folds internos = 135 entrenamientos
            # por GridSearch (vs 1.890 antes). Reducción del 93 %.
            'n_estimators':     [300, 400, 500],
            'max_depth':        [None, 10, 20],
            'min_samples_leaf': [1, 2, 5],
        },
    },

    'v4p_3_12_catboost': {
        'ruta_csv':    r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'variables':   VARIABLES_V4P_3_12,
        'etiqueta':    'etiqueta_norad_3_12',
        'calibrar':    False,
        'n_jobs_grid': 1,
        'modelo': CatBoostClassifier(
            # learning_rate e iterations fijados: constantes en todos los
            # folds del análisis previo. bagging_temperature=0 es el ganador.
            iterations=500,
            learning_rate=0.01,
            bagging_temperature=0,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'espacio': {
            # 4 × 3 = 12 combinaciones × 3 folds = 36 entrenamientos
            # (vs 864 antes). Reducción del 96 %.
            'depth':       [4, 5, 6, 7],
            'l2_leaf_reg': [1, 5, 15],
        },
    },

    'v4l_12_48_catboost': {
        'ruta_csv':    r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'variables':   VARIABLES_V4L_12_48,
        'etiqueta':    'etiqueta_norad_12_48',
        'calibrar':    False,
        'n_jobs_grid': 1,
        'modelo': CatBoostClassifier(
            iterations=500,
            learning_rate=0.01,
            bagging_temperature=0,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
        'espacio': {
            'depth':       [4, 5, 6, 7],
            'l2_leaf_reg': [1, 5, 15],
        },
    },
}


# ─────────────────────────────────────────────
# FUNCIONES: CARGA
# ─────────────────────────────────────────────

def cargar_y_preparar(config):
    df = pd.read_csv(config['ruta_csv'])
    df = df.dropna(subset=['pf_max']).copy()

    x = df[config['variables']].copy()
    x['gender'] = (x['gender'] == 'M').astype(int)

    y           = df[config['etiqueta']].astype(int).copy()
    paciente_id = df['subject_id'].copy()

    return x, y, paciente_id


# ─────────────────────────────────────────────
# FUNCIONES: DIVISIÓN A NIVEL DE PACIENTE
# ─────────────────────────────────────────────

def dividir_por_paciente(x, y, paciente_id, proporcion):
    """
    Divide a nivel de paciente, estratificado por etiqueta máxima.
    Un mismo paciente no puede aparecer en los dos grupos.
    """
    etiqueta_paciente = pd.DataFrame({
        'subject_id': paciente_id.values,
        'etiqueta':   y.values,
    }).groupby('subject_id')['etiqueta'].max().reset_index()

    pacientes_modelo, pacientes_calibracion = train_test_split(
        etiqueta_paciente['subject_id'],
        test_size=proporcion,
        random_state=RANDOM_STATE,
        stratify=etiqueta_paciente['etiqueta'],
    )

    m_modelo      = paciente_id.isin(pacientes_modelo)
    m_calibracion = paciente_id.isin(pacientes_calibracion)

    return (
        x.loc[m_modelo], y.loc[m_modelo], paciente_id.loc[m_modelo],
        x.loc[m_calibracion], y.loc[m_calibracion],
    )


# ─────────────────────────────────────────────
# FUNCIONES: GRIDSEARCH
# ─────────────────────────────────────────────

def ejecutar_gridsearch(config, x_entreno, y_entreno, grupos):
    cv_interno = StratifiedGroupKFold(
        n_splits=N_SPLITS_INTERNOS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    busqueda = GridSearchCV(
        estimator=copy.deepcopy(config['modelo']),
        param_grid=config['espacio'],
        cv=cv_interno,
        scoring='roc_auc',
        n_jobs=config['n_jobs_grid'],
        refit=True,
    )
    busqueda.fit(x_entreno, y_entreno, groups=grupos)
    return busqueda.best_estimator_, busqueda.best_params_, busqueda.best_score_


# ─────────────────────────────────────────────
# FUNCIONES: CALIBRACIÓN
# ─────────────────────────────────────────────

def buscar_mejor_calibrador(config, x_entreno, y_entreno,
                             paciente_id_entreno):
    """
    Prueba todas las combinaciones proporción × método.
    Por cada proporción lanza un GridSearch completo en el subconjunto modelo
    y calibra en el subconjunto calibración.
    Devuelve el calibrador ganador y la tabla de resultados.
    """
    resultados = []

    for proporcion in PROPORCIONES_CALIBRACION:
        x_mod, y_mod, grupos_mod, x_cal, y_cal = dividir_por_paciente(
            x_entreno, y_entreno, paciente_id_entreno, proporcion,
        )
        modelo_sub, _, _ = ejecutar_gridsearch(
            config, x_mod, y_mod, grupos_mod,
        )

        for metodo in METODOS_CALIBRACION:
            try:
                cal = calibrar_prefit(copy.deepcopy(modelo_sub), metodo, x_cal, y_cal)
                prob = cal.predict_proba(x_cal)[:, 1]
                brier = brier_score_loss(y_cal, prob)
                resultados.append({
                    'proporcion':    proporcion,
                    'metodo':        metodo,
                    'brier_score':   brier,
                    'calibrador':    cal,
                    'n_modelo':      len(x_mod),
                    'n_calibracion': len(x_cal),
                })
            except Exception as error:
                print(f'    ERROR proporcion={proporcion} metodo={metodo}: {error}')

    if not resultados:
        raise RuntimeError('Ninguna combinación de calibración funcionó.')

    ganador = min(resultados, key=lambda d: d['brier_score'])
    tabla = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'calibrador'}
        for r in resultados
    ]).sort_values('brier_score').reset_index(drop=True)

    return ganador['calibrador'], ganador['proporcion'], ganador['metodo'], tabla


def calibrar_prefit(modelo_entrenado, metodo, x_cal, y_cal):
    """
    Calibra un modelo ya entrenado. Compatible con scikit-learn antiguo
    (cv='prefit') y nuevo (FrozenEstimator).
    """
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(modelo_entrenado),
            method=metodo,
        )
    except ImportError:
        cal = CalibratedClassifierCV(
            estimator=modelo_entrenado,
            method=metodo,
            cv='prefit',
        )
    cal.fit(x_cal, y_cal)
    return cal
# ─────────────────────────────────────────────

def calcular_ece(y_real, probabilidad, n_bins=10):
    y_real      = np.asarray(y_real)
    probabilidad = np.asarray(probabilidad)
    cortes = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mascara = (probabilidad >= cortes[i]) & (probabilidad <= cortes[i + 1])
        else:
            mascara = (probabilidad >= cortes[i]) & (probabilidad < cortes[i + 1])
        if mascara.sum() == 0:
            continue
        ece += mascara.mean() * abs(y_real[mascara].mean() - probabilidad[mascara].mean())
    return ece


def calcular_metricas(y_real, probabilidad):
    return {
        'auc':    roc_auc_score(y_real, probabilidad),
        'auprc':  average_precision_score(y_real, probabilidad),
        'brier':  brier_score_loss(y_real, probabilidad),
        'logloss': log_loss(y_real, probabilidad, labels=[0, 1]),
        'ece':    calcular_ece(y_real, probabilidad),
    }


# ─────────────────────────────────────────────
# FUNCIONES: FIGURAS
# ─────────────────────────────────────────────

def guardar_curva_roc(nombre, predicciones):
    y_real = predicciones['y_real']
    prob   = predicciones['probabilidad_final']

    fpr, tpr, _ = roc_curve(y_real, prob)
    auc = roc_auc_score(y_real, prob)

    fig, eje = plt.subplots(figsize=(7, 7))
    eje.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    eje.plot([0, 1], [0, 1], 'k--', lw=1)
    eje.set_xlabel('1 - Especificidad')
    eje.set_ylabel('Sensibilidad')
    eje.set_title(f'Curva ROC — {nombre}')
    eje.legend(loc='lower right')
    eje.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_FIGURAS, f'roc_{nombre}.png')
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  ROC: {ruta}')


def guardar_curva_precision_recall(nombre, predicciones):
    y_real = predicciones['y_real']
    prob   = predicciones['probabilidad_final']

    precision, recall, _ = precision_recall_curve(y_real, prob)
    auprc  = average_precision_score(y_real, prob)
    azar   = y_real.mean()

    fig, eje = plt.subplots(figsize=(8, 6))
    eje.plot(recall, precision, color='darkorange', lw=2,
             label=f'AUPRC = {auprc:.3f}')
    eje.axhline(azar, color='gray', linestyle='--', lw=1,
                label=f'Azar = {azar:.3f}')
    eje.set_xlabel('Recall / Sensibilidad')
    eje.set_ylabel('Precision / VPP')
    eje.set_title(f'Curva Precision-Recall — {nombre}')
    eje.legend()
    eje.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_FIGURAS, f'precision_recall_{nombre}.png')
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Precision-Recall: {ruta}')


def guardar_curva_calibracion(nombre, predicciones, calibrar):
    y_real    = predicciones['y_real']
    prob_base = predicciones['probabilidad_base']
    prob_fin  = predicciones['probabilidad_final']

    prob_max = max(prob_base.max(), prob_fin.max())
    limite   = min(1.0, prob_max * 1.1)

    fig, eje = plt.subplots(figsize=(7, 7))
    eje.plot([0, limite], [0, limite], 'b--', lw=1.5, label='Perfecta')

    frac_base, media_base = calibration_curve(
        y_real, prob_base, n_bins=N_BINS_CALIBRACION, strategy=ESTRATEGIA_BINS,
    )
    eje.plot(media_base, frac_base, 'o-', color='orange', label='Sin calibrar')

    if calibrar:
        frac_fin, media_fin = calibration_curve(
            y_real, prob_fin, n_bins=N_BINS_CALIBRACION, strategy=ESTRATEGIA_BINS,
        )
        eje.plot(media_fin, frac_fin, 'o-', color='green', label='Calibrada')

    eje.set_xlim(0, limite)
    eje.set_ylim(0, limite)
    eje.set_xlabel('Probabilidad predicha')
    eje.set_ylabel('Fracción observada de positivos')
    eje.set_title(f'Curva de calibración — {nombre}')
    eje.legend()
    eje.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_FIGURAS, f'calibracion_{nombre}.png')
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Calibración: {ruta}')


def guardar_histograma(nombre, predicciones, calibrar):
    prob_base = predicciones['probabilidad_base']
    prob_fin  = predicciones['probabilidad_final']

    fig, eje = plt.subplots(figsize=(8, 5))
    eje.hist(prob_base, bins=30, alpha=0.5, color='steelblue', label='Sin calibrar')
    if calibrar:
        eje.hist(prob_fin, bins=30, alpha=0.5, color='orange', label='Calibrada')
    eje.set_xlabel('Probabilidad predicha')
    eje.set_ylabel('Frecuencia')
    eje.set_title(f'Histograma de probabilidades — {nombre}')
    eje.legend()
    eje.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(CARPETA_FIGURAS, f'histograma_{nombre}.png')
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Histograma: {ruta}')


# ─────────────────────────────────────────────
# PROCESO PRINCIPAL POR VENTANA
# ─────────────────────────────────────────────

def procesar_ventana(nombre, config):
    print(f'\n{"=" * 65}')
    print(f'  {nombre}')
    print(f'{"=" * 65}')

    x, y, paciente_id = cargar_y_preparar(config)
    print(f'  Estancias: {len(x)} | Positivos: {y.sum()} ({100*y.mean():.2f}%)')
    print(f'  Pacientes únicos: {paciente_id.nunique()}')

    cv_externa = StratifiedGroupKFold(
        n_splits=N_SPLITS_EXTERNOS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    lista_metricas_base  = []
    lista_metricas_final = []
    lista_params         = []
    lista_predicciones   = []
    lista_calibracion    = []   # solo RF

    for num_fold, (idx_entreno, idx_test) in enumerate(
        cv_externa.split(x, y, groups=paciente_id), start=1
    ):
        print(f'\n  -- Fold {num_fold}/{N_SPLITS_EXTERNOS} --')

        x_entreno = x.iloc[idx_entreno]
        y_entreno = y.iloc[idx_entreno]
        grupos_entreno = paciente_id.iloc[idx_entreno]
        x_test = x.iloc[idx_test]
        y_test = y.iloc[idx_test]

        if config['calibrar']:
            # RF: GridSearch + búsqueda calibrador dentro del entrenamiento
            print('  Buscando mejor combinación proporción × método...')
            modelo_gridsearch_base = copy.deepcopy(config['modelo'])

            calibrador, prop_ganadora, metodo_ganador, tabla_cal = \
                buscar_mejor_calibrador(
                    config,
                    x_entreno, y_entreno, grupos_entreno,
                )

            print(f'  Ganador fold {num_fold}: prop={prop_ganadora:.0%} '
                  f'metodo={metodo_ganador} '
                  f'Brier={tabla_cal.iloc[0]["brier_score"]:.5f}')

            tabla_cal['fold'] = num_fold
            lista_calibracion.append(tabla_cal)

            # Probabilidades en test
            modelo_base_fold = calibrador.estimator
            prob_base  = modelo_base_fold.predict_proba(x_test)[:, 1]
            prob_final = calibrador.predict_proba(x_test)[:, 1]

            lista_params.append({
                'fold':              num_fold,
                'proporcion_cal':    prop_ganadora,
                'metodo_cal':        metodo_ganador,
                'brier_cal_interno': tabla_cal.iloc[0]['brier_score'],
            })

        else:
            # CatBoost: solo GridSearch, sin calibrar
            print(f'  GridSearchCV interno ({N_SPLITS_INTERNOS} folds)...')
            mejor_modelo, mejores_params, mejor_auc = ejecutar_gridsearch(
                config, x_entreno, y_entreno, grupos_entreno,
            )
            print(f'  AUC interno: {mejor_auc:.4f} | params: {mejores_params}')

            prob_base  = mejor_modelo.predict_proba(x_test)[:, 1]
            prob_final = prob_base   # sin calibración

            fila_params = {'fold': num_fold, 'auc_interno': mejor_auc}
            fila_params.update(mejores_params)
            lista_params.append(fila_params)

        # Métricas
        metricas_base  = calcular_metricas(y_test, prob_base)
        metricas_final = calcular_metricas(y_test, prob_final)
        metricas_base['fold']  = num_fold
        metricas_final['fold'] = num_fold
        lista_metricas_base.append(metricas_base)
        lista_metricas_final.append(metricas_final)

        print(f'  AUC base={metricas_base["auc"]:.4f}  '
              f'AUC final={metricas_final["auc"]:.4f}  '
              f'AUPRC final={metricas_final["auprc"]:.4f}  '
              f'Brier final={metricas_final["brier"]:.5f}')

        # Predicciones
        df_pred = pd.DataFrame({
            'fold':                num_fold,
            'indice_original':     x_test.index,
            'y_real':              y_test.values,
            'probabilidad_base':   prob_base,
            'probabilidad_final':  prob_final,
        })
        lista_predicciones.append(df_pred)

    # ── Consolidar resultados ─────────────────────────────────────────
    predicciones = pd.concat(lista_predicciones, ignore_index=True)

    df_metricas_base  = pd.DataFrame(lista_metricas_base)
    df_metricas_final = pd.DataFrame(lista_metricas_final)
    df_params         = pd.DataFrame(lista_params)

    metricas_a_resumir = ['auc', 'auprc', 'brier', 'logloss', 'ece']
    resumen_final = pd.DataFrame({
        'metrica': metricas_a_resumir,
        'media':   [df_metricas_final[m].mean() for m in metricas_a_resumir],
        'std':     [df_metricas_final[m].std()  for m in metricas_a_resumir],
        'min':     [df_metricas_final[m].min()  for m in metricas_a_resumir],
        'max':     [df_metricas_final[m].max()  for m in metricas_a_resumir],
    })

    print(f'\n  RESUMEN MÉTRICAS FINALES ({nombre}):')
    print(resumen_final.to_string(index=False))

    # ── Guardar tablas ────────────────────────────────────────────────
    predicciones.to_csv(
        os.path.join(CARPETA_TABLAS, f'predicciones_{nombre}.csv'), index=False)
    df_metricas_final.to_csv(
        os.path.join(CARPETA_TABLAS, f'metricas_por_fold_{nombre}.csv'), index=False)
    resumen_final.to_csv(
        os.path.join(CARPETA_TABLAS, f'metricas_resumen_{nombre}.csv'), index=False)
    df_params.to_csv(
        os.path.join(CARPETA_TABLAS, f'mejores_parametros_{nombre}.csv'), index=False)

    if config['calibrar'] and lista_calibracion:
        pd.concat(lista_calibracion, ignore_index=True).to_csv(
            os.path.join(CARPETA_TABLAS,
                         f'seleccion_calibrador_por_fold_{nombre}.csv'),
            index=False,
        )

    # ── Guardar figuras ───────────────────────────────────────────────
    print(f'\n  Generando figuras...')
    guardar_curva_roc(nombre, predicciones)
    guardar_curva_precision_recall(nombre, predicciones)
    guardar_curva_calibracion(nombre, predicciones, config['calibrar'])
    guardar_histograma(nombre, predicciones, config['calibrar'])

    print(f'\n  Ventana {nombre} completada.')
    return resumen_final


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 65)
    print('EVALUACIÓN FINAL — CV externa 5 folds agrupada por paciente')
    print(f'Folds externos   : {N_SPLITS_EXTERNOS}')
    print(f'Folds internos   : {N_SPLITS_INTERNOS}')
    print(f'Proporciones cal.: {PROPORCIONES_CALIBRACION}  (solo RF)')
    print(f'Métodos cal.     : {METODOS_CALIBRACION}       (solo RF)')
    print('=' * 65)

    resumenes = {}
    for nombre, config in VENTANAS.items():
        resumenes[nombre] = procesar_ventana(nombre, config)

    print('\n' + '=' * 65)
    print('RESUMEN GLOBAL')
    print('=' * 65)
    for nombre, resumen in resumenes.items():
        print(f'\n{nombre}:')
        print(resumen.to_string(index=False))

    print('\nTablas → evaluacion_final/tablas/')
    print('Figuras → evaluacion_final/figuras/')
    print('=' * 65)