import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# DATASET 12h-48h
RUTA = r'C:\Users\danie\OneDrive\Escritorio\DATA\definitivo_v4l.csv'

# GUARDAR EN MISMA CARPETA DEL SCRIPT
RUTA_SALIDA = os.path.join(os.path.dirname(__file__), 'importancia_variables_rf_v4l.txt')

# CARGA
df = pd.read_csv(RUTA)
df = df.dropna(subset=['pf_max'])

# VARIABLES 
variables = [
    'anchor_age', 'gender', 'contador_estancia_uci',

    'tiene_sepsis',
    'sofa_media', 'sofa_min', 'sofa_max',
    'ventilacion_invasiva_12h',

    'lactato_media', 'lactato_min', 'lactato_max',
    'creatinina_media', 'creatinina_min', 'creatinina_max',
    'plaquetas_media', 'plaquetas_min', 'plaquetas_max',
    'bilirrubina_media', 'bilirrubina_min', 'bilirrubina_max',
    'tp_media', 'tp_min', 'tp_max',
    'gpt_media', 'gpt_min', 'gpt_max',
    'got_media', 'got_min', 'got_max',
    'pao2_media', 'pao2_min', 'pao2_max',
    'ph_media', 'ph_min', 'ph_max',
    'leucocitos_media', 'leucocitos_min', 'leucocitos_max',
    'paco2_media', 'paco2_min', 'paco2_max',
    'bicarbonato_media', 'bicarbonato_min', 'bicarbonato_max',
    'glucemia_media', 'glucemia_min', 'glucemia_max',
    'hemoglobina_media', 'hemoglobina_min', 'hemoglobina_max',

    'hr_media', 'hr_min', 'hr_max',
    'rr_media', 'rr_min', 'rr_max',
    'temp_media', 'temp_min', 'temp_max',
    'spo2_media', 'spo2_min', 'spo2_max',
    'map_media', 'map_min', 'map_max',
    'fio2_media', 'fio2_min', 'fio2_max',

    'pf_media', 'pf_min', 'pf_max',

    'gcs_media', 'gcs_min', 'gcs_max',

    'peso_kg',
    'diuresis_ml_kg_12h'
]

X = df[variables].copy()
y = df['etiqueta_norad_12_48'].astype(int)

# CODIFICACIÓN
X['gender'] = (X['gender'] == 'M').astype(int)

#MISMO RF FINETUNEADO
modelo = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    max_features=0.3,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

modelo.fit(X, y)

# IMPORTANCIAS
df_imp = pd.DataFrame({
    'variable': X.columns,
    'importancia': modelo.feature_importances_
}).sort_values(by='importancia', ascending=False)

# GUARDAR TXT
with open(RUTA_SALIDA, 'w', encoding='utf-8') as f:
    f.write("IMPORTANCIA VARIABLES RF (VENTANA 12-48h)\n")
    f.write("-------------" + "\n\n")

    for _, row in df_imp.iterrows():
        f.write(f"{row['variable']:<35} {row['importancia']:.6f}\n")

print(f"Guardado en: {RUTA_SALIDA}")