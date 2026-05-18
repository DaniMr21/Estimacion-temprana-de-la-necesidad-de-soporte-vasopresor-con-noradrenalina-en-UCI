import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── RUTA A LA CARPETA DONDE TIENES LOS CSVS ──
CARPETA_DATOS = r'C:\Users\danie\OneDrive\Escritorio\DATA'

# ── TU DICCIONARIO COMPLETO Y EXACTO (Añadidos los CSV de MIMIC correspondientes) ──
ventanas = {
    'Corto_3_12': {
        'mimic': os.path.join(CARPETA_DATOS, 'definitivo_v4p.csv'),
        'eicu': os.path.join(CARPETA_DATOS, 'eICU_3_12_definitivo.csv'),
        'vars': ['map_min', 'pf_min', 'sofa_max', 'tp_max'],
        'label': 'Corto 3-12h (CatBoost)'
    },
    'Medio_6_24': {
        'mimic': os.path.join(CARPETA_DATOS, 'definitivo_v4.csv'),
        'eicu': os.path.join(CARPETA_DATOS, 'eICU_6_24_definitivo.csv'),
        'vars': ['pf_min', 'map_min', 'diuresis_ml_kg_6h', 'hr_media', 'sofa_max', 'ventilacion_invasiva_6h'],
        'label': 'Medio 6-24h (XGBoost cal.)'
    },
    'Medio_6_24_CAT': {
        'mimic': os.path.join(CARPETA_DATOS, 'definitivo_v4.csv'),
        'eicu': os.path.join(CARPETA_DATOS, 'eICU_6_24_definitivo.csv'),
        'vars': ['pf_min', 'rr_max', 'map_min', 'diuresis_ml_kg_6h', 'ventilacion_invasiva_6h', 'hr_media', 'glucemia_min'],
        'label': 'Medio 6-24h (CatBoost)'
    },
    'Largo_12_48': {
        'mimic': os.path.join(CARPETA_DATOS, 'definitivo_v4l.csv'),
        'eicu': os.path.join(CARPETA_DATOS, 'eICU_12_48_definitivo.csv'),
        'vars': ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min', 'map_min', 'glucemia_min', 'sofa_max'],
        'label': 'Largo 12-48h (XGBoost)'
    },
    'Largo_12_48_CAT': {
        'mimic': os.path.join(CARPETA_DATOS, 'definitivo_v4l.csv'),
        'eicu': os.path.join(CARPETA_DATOS, 'eICU_12_48_definitivo.csv'),
        'vars': ['pf_min', 'bicarbonato_min', 'rr_max', 'diuresis_ml_kg_12h', 'temp_min', 'glucemia_min'],
        'label': 'Largo 12-48h (CatBoost)'
    }
}

# ── DIRECTORIO DONDE SE ESTÁ EJECUTANDO ESTE SCRIPT ──
directorio_actual = os.getcwd()

# ── GENERACIÓN DE BOXPLOTS ──
for key, info in ventanas.items():
    try:
        # Cargar los datos
        df_mimic = pd.read_csv(info['mimic'])
        df_eicu = pd.read_csv(info['eicu'])
        
        # Etiquetar
        df_mimic['Hospital'] = 'MIMIC-IV'
        df_eicu['Hospital'] = 'eICU'
        
        # Unir
        df_junto = pd.concat([df_mimic, df_eicu], ignore_index=True)
        
        # Variables de ESTE modelo concreto
        variables = info['vars']
        n_vars = len(variables)
        
        # Crear la figura adaptativa (si tiene 7 vars, la imagen es más ancha)
        fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 5))
        fig.suptitle(f"EDA Comparativo: {info['label']}", fontsize=16, fontweight='bold')
        
        # Por si solo hay 1 variable (que no pete el array de ejes)
        if n_vars == 1:
            axes = [axes]
            
        for i, var in enumerate(variables):
            # Comprobar que la variable existe por si hay algún error de tipeo
            if var in df_junto.columns:
                sns.boxplot(
                    x='Hospital', 
                    y=var, 
                    data=df_junto, 
                    ax=axes[i], 
                    palette={'MIMIC-IV': 'lightblue', 'eICU': 'lightgreen'},
                    showfliers=False # Oculta puntos extremos para ver la caja limpia
                )
                axes[i].set_title(var, fontweight='bold')
                axes[i].set_xlabel('')
                axes[i].set_ylabel('Valor')
                axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            else:
                axes[i].set_title(f"{var}\n(NO ENCONTRADA)", color='red')
                axes[i].axis('off')
                
        plt.tight_layout()
        
        # Guardar en la carpeta DONDE EJECUTAS EL CÓDIGO
        nombre_archivo = f'EDA_Boxplots_{key}.png'
        ruta_guardado = os.path.join(directorio_actual, nombre_archivo)
        
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"✅ ¡Hecho! Gráfico {key} guardado en: {ruta_guardado}")
        
        plt.close()
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: No encuentro algún archivo CSV para {key}. Revisa las rutas. Detalle: {e}")