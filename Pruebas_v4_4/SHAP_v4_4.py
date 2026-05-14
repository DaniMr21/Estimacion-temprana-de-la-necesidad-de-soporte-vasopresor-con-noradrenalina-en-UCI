import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')

# ── RUTAS Y CARPETAS ───────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
MODELOS_DIR = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS')
SHAP_DIR    = os.path.join(BASE_DIR, 'GRAFICAS_SHAP')

os.makedirs(SHAP_DIR, exist_ok=True)

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
CONFIG_VENTANAS = {
    'Corto_3_12': {
        'ruta': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'vars': {
            'RF':   ['temp_min', 'rr_max'],
            'XGB':  ['pf_min', 'map_min', 'diuresis_ml_kg_3h', 'sofa_max'],
            'LGBM': ['map_min', 'pf_min', 'sofa_max'],
            'CAT':  ['map_min', 'pf_min', 'sofa_max', 'tp_max'],
            'NB':   ['map_min', 'temp_min', 'spo2_min'],
        }
    },
    'Medio_6_24': {
        'ruta': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'vars': {
            'LR':   ['pf_min', 'diuresis_ml_kg_6h', 'rr_max', 'sofa_max',
                     'hr_media', 'ventilacion_invasiva_6h'],
            'RF':   ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'rr_max',
                     'spo2_min', 'hr_media', 'glucemia_min', 'ventilacion_invasiva_6h'],
            'XGB':  ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media',
                     'sofa_max', 'ventilacion_invasiva_6h'],
            'LGBM': ['map_min', 'pf_min', 'diuresis_ml_kg_6h', 'spo2_min',
                     'ventilacion_invasiva_6h'],
            'CAT':  ['map_min', 'pf_min', 'diuresis_ml_kg_6h', 'hr_media',
                     'rr_max', 'glucemia_min', 'ventilacion_invasiva_6h'],
            'NB':   ['pf_min', 'diuresis_ml_kg_6h', 'rr_max', 'lactato_max',
                     'sofa_max', 'hr_media', 'ventilacion_invasiva_6h'],
        }
    },
    'Largo_12_48': {
        'ruta': r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'vars': {
            'LR':   ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min',
                     'rr_max', 'map_min'],
            'RF':   ['temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max',
                     'map_min', 'glucemia_min', 'sofa_max',
                     'diuresis_ml_kg_12h', 'bilirrubina_media'],
            'XGB':  ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min',
                     'map_min', 'glucemia_min', 'sofa_max'],
            'LGBM': ['temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max',
                     'glucemia_min', 'sofa_max', 'diuresis_ml_kg_12h',
                     'map_min', 'pf_min'],
            'CAT':  ['pf_min', 'temp_min', 'diuresis_ml_kg_12h',
                     'bicarbonato_min', 'rr_max', 'glucemia_min'],
            'NB':   ['temp_min', 'pf_min', 'spo2_min', 'rr_max', 'map_min'],
        }
    }
}


def extraer_componentes(pipeline):
    """
    Devuelve (escalador_o_None, estimador_puro) desde un Pipeline de sklearn.
    Si el objeto no es un Pipeline, devuelve (None, objeto).
    """
    if not hasattr(pipeline, 'named_steps'):
        return None, pipeline

    pasos = list(pipeline.named_steps.keys())
    estimador = pipeline.named_steps[pasos[-1]]

    # Buscar escalador en los pasos anteriores
    escalador = None
    for clave in pasos[:-1]:
        paso = pipeline.named_steps[clave]
        if hasattr(paso, 'transform'):
            escalador = paso
            break

    return escalador, estimador


def calcular_shap(modelo_nombre, escalador, estimador, X_crudo):
    """
    Calcula SHAP values eligiendo el explainer adecuado.
    - RF, XGB, LGBM, CAT: TreeExplainer sobre X_crudo (no necesitan escalado)
    - LR: LinearExplainer sobre X_escalado (necesita escalado previo)
    - NB: KernelExplainer con muestra de fondo sobre X_escalado
    Devuelve (shap_values, X_para_grafica) donde X_para_grafica mantiene
    los nombres de columna originales para que el beeswarm sea legible.
    """
    if modelo_nombre in ('RF', 'XGB', 'LGBM', 'CAT'):
        # TreeExplainer es exacto y rápido para modelos de árboles.
        # No necesita escalado porque los árboles son invariantes a la escala.
        explainer   = shap.TreeExplainer(estimador)
        shap_values = explainer.shap_values(X_crudo)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # clase positiva en RF
        return shap_values, X_crudo

    # Para LR y NB hay que pasar X escalado al explainer
    if escalador is not None:
        X_escalado = pd.DataFrame(
            escalador.transform(X_crudo),
            columns=X_crudo.columns,
            index=X_crudo.index,
        )
    else:
        X_escalado = X_crudo

    if modelo_nombre == 'LR':
        # LinearExplainer es exacto para regresión logística
        explainer   = shap.LinearExplainer(
            estimador,
            X_escalado,
            feature_perturbation='interventional',
        )
        shap_values = explainer.shap_values(X_escalado)
        return shap_values, X_escalado

    # NB: KernelExplainer aproximado con muestra de fondo
    np.random.seed(42)
    fondo = shap.sample(X_escalado, min(200, len(X_escalado)))

    def predictor(datos):
        return estimador.predict_proba(datos)[:, 1]

    explainer   = shap.KernelExplainer(predictor, fondo)
    muestra     = X_escalado.sample(min(500, len(X_escalado)), random_state=42)
    shap_values = explainer.shap_values(muestra, nsamples=100)
    return shap_values, muestra


# ── BUCLE PRINCIPAL ────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')

for ventana, conf in CONFIG_VENTANAS.items():
    print(f"\nProcesando ventana: {ventana}...")

    df = pd.read_csv(conf['ruta']).dropna(subset=['pf_max'])

    for modelo_nombre, lista_vars in conf['vars'].items():
        ruta_pkl = os.path.join(
            MODELOS_DIR, f'modelo_{ventana}_{modelo_nombre}.pkl'
        )

        if not os.path.exists(ruta_pkl):
            print(f"  [Omitido] No encontrado: {ruta_pkl}")
            continue

        print(f"  -> SHAP para {modelo_nombre}...")

        X_crudo = df[lista_vars].copy()
        if 'gender' in X_crudo.columns:
            X_crudo['gender'] = (X_crudo['gender'] == 'M').astype(int)

        pipeline_cargado        = joblib.load(ruta_pkl)
        escalador, estimador    = extraer_componentes(pipeline_cargado)

        try:
            shap_values, X_grafica = calcular_shap(
                modelo_nombre, escalador, estimador, X_crudo
            )

            fig = plt.figure(figsize=(10, max(5, len(lista_vars) * 0.6 + 2)))
            shap.summary_plot(
                shap_values, X_grafica,
                plot_type='dot',
                show=False,
                color_bar=True,
            )
            plt.title(
                f'Impacto en Riesgo de Norad (SHAP)\n'
                f'Modelo: {modelo_nombre} | Ventana: {ventana}',
                fontsize=14, fontweight='bold', pad=20,
            )
            plt.tight_layout()
            ruta_guardado = os.path.join(
                SHAP_DIR, f'SHAP_{ventana}_{modelo_nombre}.png'
            )
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"     Guardado: {ruta_guardado}")

        except Exception as error:
            print(f"  [Error] {modelo_nombre}: {error}")
            plt.close()

print("\nProceso finalizado.")