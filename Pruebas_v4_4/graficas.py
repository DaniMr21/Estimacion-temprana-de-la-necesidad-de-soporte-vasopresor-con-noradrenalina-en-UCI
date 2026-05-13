import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibrationDisplay
import warnings
warnings.filterwarnings('ignore')

# ── 1. RUTAS Y CARPETAS ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
MODELOS_DIR = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS')
GRAFICAS_DIR = os.path.join(BASE_DIR, 'GRAFICAS')

os.makedirs(GRAFICAS_DIR, exist_ok=True) # Crea la carpeta si no existe

# ── 2. DICCIONARIO COMPLETO (Para saber qué variables e CSV usa cada modelo) ──
CONFIG_VENTANAS = {
    'Corto_3_12': {
        'ruta':     r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4p.csv',
        'etiqueta': 'etiqueta_norad_3_12',
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
        'etiqueta': 'etiqueta_norad_6_24',
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
        'etiqueta': 'etiqueta_norad_12_48',
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

# ── 3. BUCLE MÁGICO PARA GENERAR LAS 17 IMÁGENES ───────────────────────────────

plt.style.use('seaborn-v0_8-whitegrid')

for ventana, conf in CONFIG_VENTANAS.items():
    print(f"\nProcesando ventana: {ventana}...")
    
    # 1. Cargar el CSV correspondiente a esta ventana una sola vez
    df = pd.read_csv(conf['ruta']).dropna(subset=['pf_max'])
    y = df[conf['etiqueta']].copy()
    
    for modelo_nombre, lista_vars in conf['vars'].items():
        nombre_archivo_pkl = f'modelo_{ventana}_{modelo_nombre}.pkl'
        ruta_pkl = os.path.join(MODELOS_DIR, nombre_archivo_pkl)
        
        # Saltar si por algún casual el modelo no está guardado
        if not os.path.exists(ruta_pkl):
            print(f" ⚠️ No se encontró el modelo {nombre_archivo_pkl}, omitiendo...")
            continue
            
        print(f"  -> Generando gráfica para {modelo_nombre}...")
        
        # 2. Preparar los datos X exactos para este modelo
        X = df[lista_vars].copy()
        if 'gender' in X.columns:
            X['gender'] = (X['gender'] == 'M').astype(int)
            
        # 3. Cargar modelo y sacar probabilidades
        modelo = joblib.load(ruta_pkl)
        probabilidades = modelo.predict_proba(X)[:, 1]
        
        # 4. Crear el lienzo de 1 fila x 3 columnas (Ancho, Alto)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(f'Rendimiento Interno: {modelo_nombre} | Ventana: {ventana}', fontsize=16, fontweight='bold', y=1.05)
        
        # --- PANEL 1: CURVA ROC ---
        fpr, tpr, _ = roc_curve(y, probabilidades)
        area_roc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC curve (AUC = {area_roc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        axes[0].set_title('Curva ROC', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('1 - Especificidad')
        axes[0].set_ylabel('Sensibilidad')
        axes[0].legend(loc="lower right")
        
        # --- PANEL 2: CURVA PR (Precision-Recall) ---
        precision, recall, _ = precision_recall_curve(y, probabilidades)
        area_pr = average_precision_score(y, probabilidades)
        prevalencia = y.mean()
        axes[1].plot(recall, precision, color='#2ca02c', lw=2.5, label=f'PR curve (AUC = {area_pr:.3f})')
        axes[1].axhline(y=prevalencia, color='gray', lw=2, linestyle='--', label=f'Base ({prevalencia:.3f})')
        axes[1].set_title('Curva Precision-Recall', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Recall (Sensibilidad)')
        axes[1].set_ylabel('Precisión (VPP)')
        axes[1].legend(loc="upper right")
        
        # --- PANEL 3: CURVA DE CALIBRACIÓN ---
        display = CalibrationDisplay.from_predictions(
            y, probabilidades, n_bins=10, strategy='quantile',
            name=modelo_nombre, ax=axes[2], color='#ff7f0e', 
            linewidth=2.5, marker='o', markersize=6
        )
        axes[2].set_title('Curva de Calibración', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Probabilidad Predicha')
        axes[2].set_ylabel('Frecuencia Real')
        axes[2].legend(loc="lower right")
        
        # 5. Ajustar márgenes y GUARDAR la imagen
        plt.tight_layout()
        nombre_imagen = f'Grafica_{ventana}_{modelo_nombre}.png'
        plt.savefig(os.path.join(GRAFICAS_DIR, nombre_imagen), dpi=300, bbox_inches='tight')
        plt.close(fig) # ¡Cierra la figura en memoria para que no pete la RAM!

print("YA ESTA LOLOLOLOLO")