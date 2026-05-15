import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

# ── CONFIGURACIÓN DE RUTAS ───────────────────────────────────────────────────
BASE_DIR = r'C:\Users\danie\TFG\Pruebas_v4_4'
DATA_DIR = r'C:\Users\danie\OneDrive\Escritorio\DATA'

CONFIG = {
    'Corto': {
        'pkl': os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Corto_3_12_CAT.pkl'),
        'csv': os.path.join(DATA_DIR, 'eICU_3_12_definitivo.csv'),
        'vars': ['map_min', 'pf_min', 'sofa_max', 'tp_max']
    },
    'Medio': {
        'pkl': os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Medio_6_24_XGB.pkl'),
        'csv': os.path.join(DATA_DIR, 'eICU_6_24_definitivo.csv'),
        'vars': ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media', 'sofa_max', 'ventilacion_invasiva_6h']
    },
    'Largo': {
        'pkl': os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS', 'modelo_Largo_12_48_XGB.pkl'),
        'csv': os.path.join(DATA_DIR, 'eICU_12_48_definitivo.csv'),
        'vars': ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'map_min', 'glucemia_min', 'sofa_max']
    }
}

for nombre, conf in CONFIG.items():
    print(f"\n--- Generando SHAP para Ventana {nombre} (eICU) ---")
    
    try:
        # 1. Cargar Pipeline y datos
        pipeline = joblib.load(conf['pkl'])
        df = pd.read_csv(conf['csv']).dropna(subset=conf['vars'])
        X = df[conf['vars']]

        # ── DESEMPAQUETAR EL PIPELINE PARA SHAP ───────────────────────────────
        # Si el objeto cargado es un Pipeline de sklearn, extraemos sus partes
        if hasattr(pipeline, 'steps'):
            print("   [Pipeline detectado] Extrayendo estimador final...")
            modelo_final = pipeline.steps[-1][1]  # El modelo es el último paso
            
            # Si el pipeline tiene más pasos (como un StandardScaler), transformamos X primero
            if len(pipeline.steps) > 1:
                X_transformado = pipeline[:-1].transform(X)
                # Si la transformación devuelve un array soso de numpy, le devolvemos sus nombres de columnas
                if isinstance(X_transformado, np.ndarray):
                    X_ready = pd.DataFrame(X_transformado, columns=conf['vars'])
                else:
                    X_ready = X_transformado
            else:
                X_ready = X
        else:
            modelo_final = pipeline
            X_ready = X

        # 2. Calcular SHAP values sobre el modelo nativo y los datos preparados
        explainer = shap.TreeExplainer(modelo_final)
        shap_values = explainer.shap_values(X_ready)

        # 3. Graficar
        plt.figure(figsize=(10, 6))
        
        # Gestión de salidas según si devuelve lista (CatBoost/algunos algoritmos) o array directo
        vals_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
        
        shap.summary_plot(vals_to_plot, X_ready, show=False)
        plt.title(f"SHAP Importance - Ventana {nombre} (Validación Externa eICU)")
        
        ruta_img = os.path.join(BASE_DIR, f'shap_eicu_{nombre.lower()}.png')
        plt.savefig(ruta_img, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ ¡Logrado! Gráfico guardado en: {ruta_img}")
        
    except Exception as e:
        print(f"❌ Error en ventana {nombre}: {e}")

print("FIN")