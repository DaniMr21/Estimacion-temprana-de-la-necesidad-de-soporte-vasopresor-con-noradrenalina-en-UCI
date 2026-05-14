import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')

# 1. RUTAS Y CARPETAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
MODELOS_DIR = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS')
SHAP_DIR = os.path.join(BASE_DIR, 'GRAFICAS_SHAP')

os.makedirs(SHAP_DIR, exist_ok=True)

# 2. DICCIONARIO COMPLETO
CONFIG_VENTANAS = {
    'Corto_3_12': {
        'ruta':     r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'vars': {
            'RF':   ['temp_min', 'rr_max'],
            'XGB':  ['pf_min', 'map_min', 'diuresis_ml_kg_3h', 'sofa_max'],
            'LGBM': ['map_min', 'pf_min', 'sofa_max'],
            'CAT':  ['map_min', 'pf_min', 'sofa_max', 'tp_max'],
            'NB':   ['map_min', 'temp_min', 'spo2_min'],
        }
    },
    'Medio_6_24': {
        'ruta':     r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4.csv',
        'vars': {
            'LR':   ['pf_min', 'diuresis_ml_kg_6h', 'rr_max', 'sofa_max', 'hr_media', 'ventilacion_invasiva_6h'],
            'RF':   ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'rr_max', 'spo2_min', 'hr_media', 'glucemia_min', 'ventilacion_invasiva_6h'],
            'XGB':  ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media', 'sofa_max', 'ventilacion_invasiva_6h'],
            'LGBM': ['map_min', 'pf_min', 'diuresis_ml_kg_6h', 'spo2_min', 'ventilacion_invasiva_6h'],
            'CAT':  ['map_min', 'pf_min', 'diuresis_ml_kg_6h', 'hr_media', 'rr_max', 'glucemia_min', 'ventilacion_invasiva_6h'],
            'NB':   ['pf_min', 'diuresis_ml_kg_6h', 'rr_max', 'lactato_max', 'sofa_max', 'hr_media', 'ventilacion_invasiva_6h'],
        }
    },
    'Largo_12_48': {
        'ruta':     r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv',
        'vars': {
            'LR':   ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'map_min'],
            'RF':   ['temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'map_min', 'glucemia_min', 'sofa_max', 'diuresis_ml_kg_12h', 'bilirrubina_media'],
            'XGB':  ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'map_min', 'glucemia_min', 'sofa_max'],
            'LGBM': ['temp_min', 'spo2_min', 'bicarbonato_min', 'rr_max', 'glucemia_min', 'sofa_max', 'diuresis_ml_kg_12h', 'map_min', 'pf_min'],
            'CAT':  ['pf_min', 'temp_min', 'diuresis_ml_kg_12h', 'bicarbonato_min', 'rr_max', 'glucemia_min'],
            'NB':   ['temp_min', 'pf_min', 'spo2_min', 'rr_max', 'map_min'],
        }
    }
}

# 3. BUCLE DE GENERACION SHAP
plt.style.use('seaborn-v0_8-whitegrid')

for ventana, conf in CONFIG_VENTANAS.items():
    print(f"Procesando ventana: {ventana}...")
    
    # Cargar datos de la ventana
    df = pd.read_csv(conf['ruta']).dropna(subset=['pf_max'])
    
    for modelo_nombre, lista_vars in conf['vars'].items():
        ruta_pkl = os.path.join(MODELOS_DIR, f'modelo_{ventana}_{modelo_nombre}.pkl')
        
        if not os.path.exists(ruta_pkl):
            print(f"  [Omitido] No se encontro {ruta_pkl}")
            continue
            
        print(f"  -> Calculando SHAP para {modelo_nombre}...")
        
        # Preparar X
        X = df[lista_vars].copy()
        if 'gender' in X.columns:
            X['gender'] = (X['gender'] == 'M').astype(int)
            
        # Cargar modelo y extraer estimador puro
        modelo_cargado = joblib.load(ruta_pkl)
        try:
            modelo_puro = modelo_cargado.named_steps['modelo']
        except AttributeError:
            modelo_puro = modelo_cargado

        # Calcular valores SHAP segun la familia del algoritmo
        try:
            if modelo_nombre in ['RF', 'XGB', 'LGBM', 'CAT']:
                # Optimizacion para modelos basados en arboles
                explainer = shap.TreeExplainer(modelo_puro)
                shap_values_raw = explainer.shap_values(X)
                
                # Random Forest suele devolver una lista (una matriz por clase)
                if isinstance(shap_values_raw, list):
                    shap_values = shap_values_raw[1]
                else:
                    shap_values = shap_values_raw
                    
            else:
                # Fallback genérico para regresión logistica y Naive Bayes
                # Usamos una muestra de fondo (100) para no colapsar la memoria
                background = shap.sample(X, 100)
                explainer = shap.KernelExplainer(modelo_puro.predict_proba, background)
                shap_values_raw = explainer.shap_values(X)
                
                if isinstance(shap_values_raw, list):
                    shap_values = shap_values_raw[1]
                else:
                    shap_values = shap_values_raw

            # Dibujar y guardar la grafica
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type="dot", show=False, color_bar=True)
            
            plt.title(f'Impacto en Riesgo de Norad (SHAP)\nModelo: {modelo_nombre} | Ventana: {ventana}', 
                      fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            nombre_archivo = f'SHAP_{ventana}_{modelo_nombre}.png'
            ruta_guardado = os.path.join(SHAP_DIR, nombre_archivo)
            plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"  [Error] Fallo al generar SHAP para {modelo_nombre}: {e}")
            plt.close()

print("\nProceso finalizado. Graficas SHAP generadas en el directorio.")