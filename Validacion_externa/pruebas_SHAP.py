import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# ── 1. 
class ModeloCalibrado:
    def __init__(self, modelo_base, calibrador, tipo_calibrador):
        self.modelo_base     = modelo_base
        self.calibrador      = calibrador
        self.tipo_calibrador = tipo_calibrador

    def predict_proba(self, X):
        prob_bruta = self.modelo_base.predict_proba(X)[:, 1]
        if isinstance(self.calibrador, IsotonicRegression):
            prob_cal = self.calibrador.predict(prob_bruta)
        else:
            prob_cal = self.calibrador.predict_proba(
                prob_bruta.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - prob_cal, prob_cal])

    def predict(self, X, umbral=0.5):
        return (self.predict_proba(X)[:, 1] >= umbral).astype(int)


# ── 2. EL PELADOR DE CAPAS DEFINITIVO ────────────────────────────────────────
def extraer_arbol_y_datos(modelo_cargado, X_inicial, columnas):
    """
    Bucle infinito que pela todas las capas del modelo (ModeloCalibrado, 
    Pipelines, etc.) hasta llegar al algoritmo de árboles, transformando 
    los datos X por el camino si hay Scalers.
    """
    X_actual = X_inicial.copy()
    modelo = modelo_cargado
    
    while True:
        # Capa 1: Si es TU clase calibrada, sacamos el modelo base
        if isinstance(modelo, ModeloCalibrado):
            modelo = modelo.modelo_base
            
        # Capa 2: Si es un Pipeline de Sklearn, transformamos datos y sacamos el final
        elif hasattr(modelo, 'steps'):
            if len(modelo.steps) > 1:
                X_transformado = modelo[:-1].transform(X_actual)
                if isinstance(X_transformado, np.ndarray):
                    X_actual = pd.DataFrame(X_transformado, columns=columnas)
                else:
                    X_actual = X_transformado
            modelo = modelo.steps[-1][1]
            
        # Capa 3: Si usaste CalibratedClassifierCV oficial
        elif hasattr(modelo, 'calibrated_classifiers_'):
            modelo = modelo.calibrated_classifiers_[0].estimator
            
        # Si no tiene ninguna de estas capas... ¡Hemos llegado al núcleo!
        else:
            break
            
    return modelo, X_actual


# ── 3. CONFIGURACIÓN DE RUTAS ────────────────────────────────────────────────
BASE_DIR = r'C:\Users\danie\TFG\Pruebas_v4_4'
DATA_DIR = r'C:\Users\danie\OneDrive\Escritorio\DATA'

CONFIG = {
    'Corto_CAT': {
        'pkl': os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Corto_3_12_CAT.pkl'),
        'csv': os.path.join(DATA_DIR, 'eICU_3_12_definitivo.csv'),
        'vars': ['map_min', 'pf_min', 'sofa_max', 'tp_max']
    },
    'Medio_XGB_Cal': {
        'pkl': os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Medio_6_24_XGB_calibrado.pkl'),
        'csv': os.path.join(DATA_DIR, 'eICU_6_24_definitivo.csv'),
        'vars': ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media', 'sofa_max', 'ventilacion_invasiva_6h']
    },
    'Medio_CAT': {
        'pkl': os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Medio_6_24_CAT.pkl'),
        'csv': os.path.join(DATA_DIR, 'eICU_6_24_definitivo.csv'),
        'vars': ['pf_min', 'rr_max', 'map_min', 'diuresis_ml_kg_6h', 'ventilacion_invasiva_6h', 'hr_media', 'glucemia_min']
    },
    'Largo_XGB': {
        'pkl': os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Largo_12_48_XGB.pkl'),
        'csv': os.path.join(DATA_DIR, 'eICU_12_48_definitivo.csv'),
        'vars': ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'map_min', 'glucemia_min', 'sofa_max']
    },
    'Largo_CAT': {
        'pkl': os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Largo_12_48_CAT.pkl'),
        'csv': os.path.join(DATA_DIR, 'eICU_12_48_definitivo.csv'),
        'vars': ['pf_min', 'bicarbonato_min', 'rr_max', 'diuresis_ml_kg_12h', 'temp_min', 'glucemia_min']
    }
}

# ── 4. BUCLE PRINCIPAL ───────────────────────────────────────────────────────
for nombre, conf in CONFIG.items():
    print(f"\n--- Generando SHAP para {nombre} (eICU) ---")
    
    try:
        # Cargar datos crudos y modelo
        pipeline_crudo = joblib.load(conf['pkl'])
        df = pd.read_csv(conf['csv']).dropna(subset=conf['vars'])
        X = df[conf['vars']]

        # Mágia: Pela todas las capas (tu clase -> Pipeline -> XGBoost)
        modelo_puro, X_procesado = extraer_arbol_y_datos(pipeline_crudo, X, conf['vars'])
        print(f"  [+] Corazón extraído para SHAP: {type(modelo_puro).__name__}")

        # Calcular SHAP values
        explainer = shap.TreeExplainer(modelo_puro)
        shap_values = explainer.shap_values(X_procesado)

        # Graficar
        plt.figure(figsize=(10, 6))
        vals_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
        
        shap.summary_plot(vals_to_plot, X_procesado, show=False)
        plt.title(f"SHAP Importance - {nombre} (Validación Externa eICU)")
        
        ruta_img = os.path.join(BASE_DIR, f'shap_eicu_{nombre.lower()}.png')
        plt.savefig(ruta_img, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ ¡Logrado! Gráfico guardado en: {ruta_img}")
        
    except Exception as e:
        print(f"❌ Error en ventana {nombre}: {e}")

print("\nFIN")