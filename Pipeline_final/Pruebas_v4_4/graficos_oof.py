import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
MODELOS_DIR = os.path.join(BASE_DIR, 'MODELOS_ENTRENADOS')
GRAFICAS_DIR = os.path.join(BASE_DIR, 'GRAFICAS_HONESTAS_OOF') # Carpeta nueva

os.makedirs(GRAFICAS_DIR, exist_ok=True)
COLUMNA_ID = 'subject_id'


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

plt.style.use('seaborn-v0_8-whitegrid')

for ventana, conf in CONFIG_VENTANAS.items():
    print(f"\nProcesando ventana: {ventana}...")
    
    # Cargar datos
    df = pd.read_csv(conf['ruta'])
    y = df[conf['etiqueta']].copy()
    ids = df[COLUMNA_ID].copy()
    
    # El mismo KFold de la CV externa
    cv_externo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    for modelo_nombre, lista_vars in conf['vars'].items():
        ruta_pkl = os.path.join(MODELOS_DIR, f'modelo_{ventana}_{modelo_nombre}.pkl')
        if not os.path.exists(ruta_pkl):
            continue
            
        print(f"Extrayendo probabilidades OOF para {modelo_nombre}...")
        
        # Preparar X
        X = df[lista_vars].copy()
        if 'gender' in X.columns:
            X['gender'] = (X['gender'] == 'M').astype(int)
            
        # Cargar modelo (ya viene con los mejores hiperparámetros)
        modelo = joblib.load(ruta_pkl)
        
        #Calcular probabilidades Out-of-Fold (Honestas)
        probs_oof = cross_val_predict(
            estimator=modelo, 
            X=X, 
            y=y, 
            groups=ids, 
            cv=cv_externo, 
            method='predict_proba'
        )[:, 1]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(f'Rendimiento Validación Cruzada (OOF): {modelo_nombre} | Ventana: {ventana}', fontsize=16, fontweight='bold', y=1.05)
        
        # 1. ROC
        fpr, tpr, _ = roc_curve(y, probs_oof)
        area_roc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC OOF (AUC = {area_roc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        axes[0].set_title('Curva ROC', fontsize=13, fontweight='bold')
        axes[0].legend(loc="lower right")
        
        # 2. PR
        precision, recall, _ = precision_recall_curve(y, probs_oof)
        area_pr = average_precision_score(y, probs_oof)
        axes[1].plot(recall, precision, color='#2ca02c', lw=2.5, label=f'PR OOF (AUC = {area_pr:.3f})')
        axes[1].axhline(y=y.mean(), color='gray', lw=2, linestyle='--')
        axes[1].set_title('Curva Precision-Recall', fontsize=13, fontweight='bold')
        axes[1].legend(loc="upper right")
        
        # 3. Calibración
        display = CalibrationDisplay.from_predictions(
            y, probs_oof, n_bins=10, strategy='quantile',
            name=modelo_nombre, ax=axes[2], color='#ff7f0e', 
            linewidth=2.5, marker='o', markersize=6
        )
        axes[2].set_title('Curva de Calibración', fontsize=13, fontweight='bold')
        axes[2].legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFICAS_DIR, f'Grafica_OOF_{ventana}_{modelo_nombre}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

print("fin, ya están las oof")