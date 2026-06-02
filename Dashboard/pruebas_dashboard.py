import os
import io
import joblib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

class ModeloCalibrado:
    """Wrapper necesario para deserializar el pkl del Medio calibrado."""
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

CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))

MODELOS = {
    'Medio': {
        'modelo_pkl' : 'modelo_Medio_6_24_XGB_calibrado.pkl',
        'calibrador' : None,
        'vars'       : ['pf_min', 'map_min', 'diuresis_ml_kg_6h',
                        'hr_media', 'sofa_max', 'ventilacion_invasiva_6h'],
        'etiquetas_var': {
            'pf_min'                 : 'PF mín (PaO₂/FiO₂)',
            'map_min'                : 'MAP mín (mmHg)',
            'diuresis_ml_kg_6h'      : 'Diuresis (ml/kg/6h)',
            'hr_media'               : 'FC media (lpm)',
            'sofa_max'               : 'SOFA máx',
            'ventilacion_invasiva_6h': 'VM invasiva',
        },
        'ventana': '6-24h',
    },
    'Largo': {
        'modelo_pkl' : 'modelo_Largo_12_48_XGB.pkl',
        'calibrador' : None,
        'vars'       : ['temp_min', 'pf_min', 'spo2_min', 'bicarbonato_min',
                        'map_min', 'glucemia_min', 'sofa_max'],
        'etiquetas_var': {
            'temp_min'       : 'Temp mín (°C)',
            'pf_min'         : 'PF mín (PaO₂/FiO₂)',
            'spo2_min'       : 'SpO₂ mín (%)',
            'bicarbonato_min': 'Bicarbonato mín',
            'map_min'        : 'MAP mín (mmHg)',
            'glucemia_min'   : 'Glucemia mín',
            'sofa_max'       : 'SOFA máx',
        },
        'ventana': '12-48h',
    },
}

DEMO_CSV = {
    'Demo 1 — Paciente estable (riesgo bajo esperado)': """\
NOMBRE,APELLIDO 1,APELLIDO 2,ID,SOFA,MAP,TP,FiO2,PaO2,Diuresis_mL_kg,Frec Cardiaca,Vent. Invasiva,SpO2,Bicarbonato,Temperatura,Glucemia
Uno,Demo,Prueba,DEMO-001,2,82,14.0,0.30,120,0.55,78,0,97,24,37.1,108
Uno,Demo,Prueba,DEMO-001,2,79,14.0,0.30,118,0.48,80,0,97,23,36.9,112
Uno,Demo,Prueba,DEMO-001,2,84,13.5,0.30,122,0.52,76,0,98,24,37.0,105
Uno,Demo,Prueba,DEMO-001,2,81,14.0,0.30,119,0.50,79,0,97,23,36.8,110
""",
    'Demo 2 — Paciente con deterioro hemodinámico (riesgo alto esperado)': """\
NOMBRE,APELLIDO 1,APELLIDO 2,ID,SOFA,MAP,TP,FiO2,PaO2,Diuresis_mL_kg,Frec Cardiaca,Vent. Invasiva,SpO2,Bicarbonato,Temperatura,Glucemia
Dos,Demo,Test,DEMO-002,9,58,18.5,0.70,98,0.12,112,1,91,16,38.6,185
Dos,Demo,Test,DEMO-002,10,54,19.0,0.75,92,0.10,118,1,89,15,38.8,192
Dos,Demo,Test,DEMO-002,10,51,19.5,0.80,88,0.08,122,1,88,14,39.0,198
Dos,Demo,Test,DEMO-002,11,48,20.0,0.85,84,0.07,128,1,87,13,39.2,205
""",
}

@st.cache_resource
def cargar_modelo(nombre):
    cfg = MODELOS[nombre]
    ruta = os.path.join(CARPETA_BASE, cfg['modelo_pkl'])
    try:
        modelo = joblib.load(ruta)
    except Exception:
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
    return modelo, None


@st.cache_resource
def cargar_referencia(nombre):
    ruta_ref = os.path.join(
        CARPETA_BASE, f'referencia_percentiles_{nombre.lower()}.npy')
    if os.path.exists(ruta_ref):
        return np.load(ruta_ref)
    st.error(f'No se encontró el fichero de referencia: {ruta_ref}')
    st.stop()

def agregar_analitica(df_raw, nombre_modelo):
    """Convierte CSV bruto del paciente en vector de variables del modelo."""
    pf = (df_raw['PaO2'] / df_raw['FiO2']).replace([np.inf, -np.inf], np.nan)
    valores = {
        'pf_min'                 : pf.min(),
        'map_min'                : df_raw['MAP'].min(),
        'diuresis_ml_kg_6h'      : df_raw['Diuresis_mL_kg'].sum(),
        'hr_media'               : df_raw['Frec Cardiaca'].mean(),
        'sofa_max'               : df_raw['SOFA'].max(),
        'ventilacion_invasiva_6h': int(df_raw['Vent. Invasiva'].max()),
        'temp_min'               : df_raw['Temperatura'].min(),
        'spo2_min'               : df_raw['SpO2'].min(),
        'bicarbonato_min'        : df_raw['Bicarbonato'].min(),
        'glucemia_min'           : df_raw['Glucemia'].min(),
        'tp_max'                 : df_raw['TP'].max(),
    }
    vars_modelo = MODELOS[nombre_modelo]['vars']
    fila = {v: valores[v] for v in vars_modelo}
    return pd.DataFrame([fila])

def extraer_estimador(modelo):
    m = modelo
    if hasattr(m, 'modelo_base'):
        m = m.modelo_base
    if hasattr(m, 'steps'):
        m = m.steps[-1][1]
    return m


def calcular_shap(modelo, X_fila, vars_modelo):
    estimador = extraer_estimador(modelo)
    explainer = shap.TreeExplainer(estimador)
    sv = explainer.shap_values(X_fila)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    vals = sv[0] if sv.ndim == 2 else sv
    return dict(zip(vars_modelo, vals))

def calcular_percentil(prob, distribucion_referencia):
    rango = np.searchsorted(distribucion_referencia, prob, side='right')
    return int(round(100 * rango / len(distribucion_referencia)))

def mostrar_resultados(df_raw, nombre_modelo):
    cfg = MODELOS[nombre_modelo]
    try:
        X_fila = agregar_analitica(df_raw, nombre_modelo)
        modelo, calibrador = cargar_modelo(nombre_modelo)
        referencia = cargar_referencia(nombre_modelo)

        prob = modelo.predict_proba(X_fila)[:, 1][0]
        if calibrador is not None:
            prob = calibrador.predict_proba(np.array([[prob]]))[:, 1][0]

        percentil = calcular_percentil(prob, referencia)
        shap_dict = calcular_shap(modelo, X_fila, cfg['vars'])

        col_izq, col_der = st.columns([1, 1.4])

        with col_izq:
            st.markdown('**PERCENTIL DE RIESGO**')
            st.markdown(
                f'<div class="percentil-grande">{percentil}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div style="border:1px solid #ccc; padding:10px; '
                'font-size:12px; background:#f9f9f9; margin-top:10px;">'
                '<b>¿Qué significa este valor?</b><br><br>'
                'El percentil indica la posición del paciente dentro de la '
                'distribución de probabilidades obtenida al aplicar el modelo '
                'sobre la cohorte de referencia MIMIC-IV. Por ejemplo, un valor de 70 '
                'significa que el 70&nbsp;% de los pacientes de esa cohorte '
                'obtuvieron una probabilidad estimada inferior a la de este '
                'paciente. No es una probabilidad directa de que ocurra el '
                'evento, sino una medida relativa de riesgo dentro de una '
                'población UCI de referencia.'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown('<br>', unsafe_allow_html=True)
            if 'ID' in df_raw.columns:
                pid = df_raw['ID'].iloc[0]
                nombre = ' '.join(str(df_raw[c].iloc[0])
                                  for c in ['NOMBRE', 'APELLIDO 1', 'APELLIDO 2']
                                  if c in df_raw.columns)
                st.markdown(
                    f'**Paciente:** {nombre}  \n'
                    f'**ID:** {pid}  \n'
                    f'**N medidas:** {len(df_raw)}'
                )

        with col_der:
            st.markdown('**FACTORES DE RIESGO (SHAP)**')
            items = sorted(shap_dict.items(), key=lambda x: x[1])
            etiquetas = [cfg['etiquetas_var'].get(k, k) for k, _ in items]
            valores   = [v for _, v in items]
            colores   = ['#2BFF00' if v < 0 else '#FF0000' for v in valores]

            fig, ax = plt.subplots(figsize=(7, max(3, len(items) * 0.5)))
            fig.patch.set_facecolor('white')
            ax.barh(etiquetas, valores, color=colores,
                    edgecolor='black', linewidth=1)
            ax.axvline(0, color='black', lw=1)
            ax.set_xlabel('← protege         empuja →')
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(colors='black')
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_color('black')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption('Rojo = empuja al riesgo · Verde = protege')

            st.markdown(
                '<div style="border:1px solid #ccc; padding:8px; '
                'font-size:12px; background:#f9f9f9; margin-top:6px;">'
                '<b>¿Cómo leer este gráfico?</b><br>'
                'Cada barra muestra cuánto contribuye esa variable a la '
                'predicción <b>de este paciente concreto</b>. '
                'Las barras rojas empujan el riesgo estimado hacia arriba; '
                'las verdes lo reducen. '
                'La longitud indica la magnitud de la contribución. '
                'Esta explicación es local: refleja el perfil individual '
                'del paciente, no la importancia global de la variable '
                'en el modelo.'
                '</div>',
                unsafe_allow_html=True
            )

    except KeyError as e:
        st.error(f'Falta columna en el CSV: {e}')
    except Exception as e:
        st.error(f'Error procesando los datos: {e}')

st.set_page_config(page_title='Dashboard NORAD UCI', layout='wide')

st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; font-family: 'Courier New', monospace; }
    h1, h2, h3, h4 { color: #000000; font-family: 'Courier New', monospace; }
    .titulo-principal {
        text-align: center; padding: 10px; border: 2px solid black;
        margin-bottom: 5px; font-size: 18px; font-weight: bold;
    }
    .aviso-prototipo {
        background-color: #ff8c00; color: #ffffff; padding: 12px;
        text-align: center; border: 2px solid black; margin: 10px 0;
        font-weight: bold; font-size: 13px;
    }
    .panel-borde { border: 1px solid black; padding: 12px; margin: 5px 0; background: #ffffff; }
    .percentil-grande {
        font-size: 90px; font-weight: bold; text-align: center; line-height: 1;
        font-family: 'Courier New', monospace;
    }
    .percentil-etiqueta { text-align: center; font-size: 14px; margin-top: 5px; }
    .pie { font-size: 11px; color: #555; text-align: right; padding-top: 8px; }
    div.stButton > button {
        width: 100%; background-color: #ffffff; color: #000000;
        border: 2px solid #000000; border-radius: 0; height: 50px;
        font-family: 'Courier New', monospace; font-weight: bold;
    }
    div.stButton > button:hover { background-color: #000000; color: #ffffff; }
    .stFileUploader label { color: #000000; font-family: 'Courier New', monospace; }
    div[role="radiogroup"] label { color: #000000 !important; }
    div[role="radiogroup"] label p { color: #000000 !important; }
    .stSelectbox label { color: #000000 !important; }
    .stSelectbox label p { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="titulo-principal">DASHBOARD DE ESTIMACIÓN TEMPRANA'
    'DE LA NECESIDAD DE INICIO DE NORADRENALINA EN UCI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="aviso-prototipo">'
    '⚠ PROTOTIPO DE INVESTIGACIÓN — NO USAR BAJO NINGÚN CONCEPTO EN PRÁCTICA '
    'CLÍNICA REAL ⚠<br>'
    'Sin validación clínica ni autorización regulatoria. '
    'Las decisiones clínicas son responsabilidad exclusiva del facultativo.'
    '</div>',
    unsafe_allow_html=True
)

for clave, defecto in [
    ('modelo_sel', None),
    ('mostrar_manual', False),
    ('modo_demo', False),
]:
    if clave not in st.session_state:
        st.session_state[clave] = defecto

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button('VENTANA MEDIA  (6-24h)'):
        st.session_state.modelo_sel     = 'Medio'
        st.session_state.mostrar_manual = False
        st.session_state.modo_demo      = False
with c2:
    if st.button('VENTANA LARGA  (12-48h)'):
        st.session_state.modelo_sel     = 'Largo'
        st.session_state.mostrar_manual = False
        st.session_state.modo_demo      = False
with c3:
    if st.button('MODO DEMO'):
        st.session_state.modo_demo      = True
        st.session_state.mostrar_manual = False
        st.session_state.modelo_sel     = None
with c4:
    if st.button('MANUAL / INFO'):
        st.session_state.mostrar_manual = True
        st.session_state.modo_demo      = False

st.markdown('---')

#MODO DEMO

if st.session_state.modo_demo:
    st.markdown('### MODO DEMO')
    st.info(
        'Los datos mostrados son **ficticios** y se utilizan únicamente para '
        'ilustrar el funcionamiento del dashboard. No corresponden a ningún '
        'paciente real.'
    )

    demo_nombre = st.selectbox(
        'Selecciona un caso de demostración:',
        list(DEMO_CSV.keys())
    )

    ventana_demo = st.radio(
        'Ventana temporal:',
        ['Medio (6-24h)', 'Largo (12-48h)'],
        horizontal=True
    )
    nombre_modelo_demo = 'Medio' if ventana_demo.startswith('Medio') else 'Largo'

    with st.expander('Ver datos del caso de demo'):
        df_demo_preview = pd.read_csv(io.StringIO(DEMO_CSV[demo_nombre]))
        st.dataframe(df_demo_preview, use_container_width=True)

    st.markdown('---')

    df_demo = pd.read_csv(io.StringIO(DEMO_CSV[demo_nombre]))
    cfg_demo = MODELOS[nombre_modelo_demo]
    st.markdown(
        f'**Modelo activo:** Ventana {nombre_modelo_demo} '
        f'({cfg_demo["ventana"]}) — caso: _{demo_nombre}_'
    )
    mostrar_resultados(df_demo, nombre_modelo_demo)

elif st.session_state.mostrar_manual:
    st.markdown('### MANUAL DE USO')
    st.markdown("""
**Objetivo.** Estimar de manera temprana la probabilidad de la necesidad de soporte vasopresor con noradrenalina en UCI

**Ventanas disponibles.**
- *Medio:* probabilidad de inicio entre 6 y 24 horas a partir del momento actual.
- *Largo:* probabilidad de inicio entre 12 y 48 horas a partir del momento actual.

**Modo de uso.**
1. Seleccione la ventana temporal con uno de los botones superiores.
2. Cargue un CSV con las medidas del paciente (ver formato abajo) entre las 0 y 6 horas desde el ingreso o entre las 0 y 12 horas según corresponda.
3. Consulte el percentil de riesgo y los factores SHAP en el panel de resultados.

**Formato del CSV.** Una fila por medición. Columnas obligatorias:
`NOMBRE, APELLIDO 1, APELLIDO 2, ID, SOFA, MAP, TP, FiO2, PaO2,
Diuresis_mL_kg, Frec Cardiaca, Vent. Invasiva, SpO2, Bicarbonato,
Temperatura, Glucemia`

**Interpretación del percentil.**
El valor (1–100) indica la posición del paciente dentro de la distribución
de probabilidades obtenida sobre la cohorte de referencia MIMIC-IV.
Un percentil de 90 significa que el 90 % de los pacientes de esa cohorte
obtuvieron una probabilidad estimada inferior a la de este paciente.
No es una probabilidad directa de evento, sino una medida relativa de riesgo
dentro de una población UCI de referencia.

**Interpretación SHAP.**
El gráfico de barras muestra la contribución individual de cada variable
a la predicción de *este paciente concreto*. Las barras rojas empujan el
riesgo hacia arriba; las verdes lo reducen. Es una explicación local,
no una medida de importancia global del modelo.

**Modo Demo.** Disponible en el botón superior para explorar el dashboard
con dos casos ficticios sin necesidad de cargar un CSV real.

**Aviso.** Prototipo académico sin validación clínica prospectiva ni
autorización regulatoria. Cualquier decisión clínica debe basarse
exclusivamente en la valoración del facultativo responsable.
    """)

#MODO NORMAL (CSV REAL)

elif st.session_state.modelo_sel is None:
    st.info('Seleccione una ventana temporal o pulse MODO DEMO para comenzar.')

else:
    cfg = MODELOS[st.session_state.modelo_sel]
    st.markdown(
        f'**Modelo activo:** Ventana {st.session_state.modelo_sel} '
        f'({cfg["ventana"]})'
    )

    archivo = st.file_uploader('Cargar analítica del paciente (CSV)', type=['csv'])

    if archivo is None:
        st.info('Cargue un CSV con el formato indicado en el manual.')
    else:
        df_raw = pd.read_csv(archivo)
        mostrar_resultados(df_raw, st.session_state.modelo_sel)

st.markdown('---')
st.markdown(
    f'<div style="text-align:right; font-family: Courier New, monospace; '
    f'font-size:15px; color:#000; padding-top:8px;">'
    f'Última actualización: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</div>',
    unsafe_allow_html=True
)