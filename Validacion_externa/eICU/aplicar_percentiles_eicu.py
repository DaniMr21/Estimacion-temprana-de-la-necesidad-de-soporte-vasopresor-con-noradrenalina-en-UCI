"""
Paso 2: Aplicar los percentiles de MIMIC a los datos de eICU.

Los umbrales de winsorización vienen de MIMIC (entrenamiento).
Aplicarlos a eICU garantiza que no hay fuga de información del test set.

Si una variable de eICU no estaba en MIMIC (no tiene percentiles guardados),
se avisa pero NO se winsorizan esos valores.
"""

import os
import pandas as pd
import numpy as np

CARPETA_EICU   = r'C:\Users\danie\OneDrive\Escritorio\DATA'
CARPETA_PERC   = r'C:\Users\danie\OneDrive\Escritorio\DATA'  # donde guardó el paso 1

VENTANAS = {
    'v4p': {
        'entrada'    : os.path.join(CARPETA_EICU, 'eICU_3_12.csv'),
        'salida'     : os.path.join(CARPETA_EICU, 'eICU_3_12_definitivo.csv'),
        'percentiles': os.path.join(CARPETA_PERC, 'percentiles_mimic_v4p.csv'),
        'etiqueta'   : 'etiqueta_norad_3_12',
        'excluir_extra': [],
    },
    'v4': {
        'entrada'    : os.path.join(CARPETA_EICU, 'eICU_6_24.csv'),
        'salida'     : os.path.join(CARPETA_EICU, 'eICU_6_24_definitivo.csv'),
        'percentiles': os.path.join(CARPETA_PERC, 'percentiles_mimic_v4.csv'),
        'etiqueta'   : 'etiqueta_norad_6_24',
        'excluir_extra': [],
    },
    'v4l': {
        'entrada'    : os.path.join(CARPETA_EICU, 'eICU_12_48.csv'),
        'salida'     : os.path.join(CARPETA_EICU, 'eICU_12_48_definitivo.csv'),
        'percentiles': os.path.join(CARPETA_PERC, 'percentiles_mimic_v4l.csv'),
        'etiqueta'   : 'etiqueta_norad_12_48',
        'excluir_extra': ['ventilacion_invasiva_12h'],
    },
}

EXCLUIR_BASE = [
    'subject_id', 'stay_id',
    'anchor_age', 'gender', 'peso_kg',
    'contador_estancia_uci',
    'horas_hasta_norad',
    'gcs_min',
    'sofa_max',
]

for nombre, config in VENTANAS.items():
    print(f'\n{"=" * 55}')
    print(f'  {nombre} — aplicando percentiles de MIMIC a eICU')
    print(f'{"=" * 55}')

    df          = pd.read_csv(config['entrada'])
    df_perc     = pd.read_csv(config['percentiles'])
    umbrales    = dict(zip(df_perc['variable'],
                           zip(df_perc['p2'], df_perc['p98'])))

    excluir     = EXCLUIR_BASE + [config['etiqueta']] + config['excluir_extra']
    cols_eicu   = [c for c in df.columns
                   if c not in excluir and pd.api.types.is_numeric_dtype(df[c])]

    print(f'  Filas eICU: {len(df)}')
    print(f'  Positivos:  {int(df[config["etiqueta"]].sum())}  '
          f'({100 * df[config["etiqueta"]].mean():.2f}%)')
    print()

    total_bajos = 0
    total_altos = 0
    sin_umbral  = []

    for col in cols_eicu:
        if col not in umbrales:
            sin_umbral.append(col)
            continue

        p2, p98 = umbrales[col]
        serie   = df[col].astype(float)
        n_bajo  = int((serie < p2).sum())
        n_alto  = int((serie > p98).sum())

        df[col] = serie.clip(lower=p2, upper=p98)

        total_bajos += n_bajo
        total_altos += n_alto

        if n_bajo > 0 or n_alto > 0:
            print(f'  {col:35s}  clip [{p2:.3f}, {p98:.3f}]  '
                  f'↓{n_bajo}  ↑{n_alto}')

    print()
    print(f'  Valores recortados por abajo : {total_bajos}')
    print(f'  Valores recortados por arriba: {total_altos}')

    if sin_umbral:
        print(f'\n  AVISO — variables en eICU sin percentiles de MIMIC '
              f'(no se winsorizan): {sin_umbral}')

    df.to_csv(config['salida'], index=False)
    print(f'\n  Guardado en: {config["salida"]}')

print('\n[Fin] eICU winsorizado con umbrales de MIMIC.')
