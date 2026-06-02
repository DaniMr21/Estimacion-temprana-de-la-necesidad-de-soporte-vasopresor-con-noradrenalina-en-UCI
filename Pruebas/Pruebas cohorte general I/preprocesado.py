import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split

def crear_engine():
    contrasena = quote_plus("12345678")
    motor = create_engine(
        f"postgresql+psycopg2://postgres:{contrasena}@localhost:5432/MIMIC-IV"
    )
    return motor

def cargar_dataset():
    consulta = "select * from public.dataset_modelo_6h"
    motor = crear_engine()
    dataframe = pd.read_sql(consulta, motor)
    return dataframe

def preparar_datos(dataframe: pd.DataFrame):
    target = "etiqueta_norad"
    fuera = [      #Esto ya no nos sirve pq cada fila ya es un pac diferente y ha pasado los filtros de la cohorte inicial planteada. No tiene que entrar al modelo 
        "stay_id",
        "subject_id",
        "hadm_id",
        "intime",
        "outtime",
        "rn",
        "horas_uci",
        "inicio_noradrenalina",
        "horas_hasta_norad",
    ]

    x = dataframe.drop(columns = fuera + [target]).copy() #vars que son predictoras / copy para no romper nada
    y = dataframe[target].copy() #objetivo 

    return x, y

def mostrar_resumen(x: pd.DataFrame, y: pd.Series):
    print(x.shape)
    print(x.columns.tolist())
    print(y.value_counts(dropna=False))
    print(y.mean())
    print(x.isnull().sum().sort_values(ascending=False))
    print("//////////////////////////////////////////////////////")

#-------uepa

def dividir_train_test(x: pd.DataFrame, y: pd.Series):
    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        x,
        y,
        test_size=0.2, #80% entrenar y 20% evaluar 
        random_state=42,
        stratify=y, #manentener positivos de norad equilibrados en train y test
    )
    return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba

def main():
    dataframe = cargar_dataset()
    x, y = preparar_datos(dataframe)
    mostrar_resumen(x, y)

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = dividir_train_test(x, y)

    print("shape entrenamiento:")
    print(x_entrenamiento.shape, y_entrenamiento.shape)

    print("shape prueba:")
    print(x_prueba.shape, y_prueba.shape)

    print("positivos entrenamiento:")
    print(y_entrenamiento.mean())

    print("positivos prueba:")
    print(y_prueba.mean())

if __name__ == "__main__":
    main()

#Todo en orden por aquí, lo malo que la prevalencia es 4% aprox