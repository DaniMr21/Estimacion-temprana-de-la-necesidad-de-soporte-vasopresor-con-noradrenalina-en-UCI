import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def cargar_datos():
    contrasena = quote_plus("12345678")
    engine = create_engine(f"postgresql+psycopg2://postgres:{contrasena}@localhost:5432/MIMIC-IV")
    df = pd.read_sql("select * from public.dataset_modelo_6h", engine)
    return df

#Esto se queda comentado porque no sube el AUC, lo baja, no meter variables derivadas
"""""""""""""""
def preparar(df):
    target = "etiqueta_norad"
    fuera = ["stay_id", "subject_id", "hadm_id","intime", "outtime", "rn", "horas_uci", "inicio_noradrenalina", "horas_hasta_norad"]
    x = df.drop(columns=fuera + [target]).copy()
    y = df[target].copy()

    x["delta_card"] = x["max_card"] - x["min_card"]
    x["delta_map"] = x["map_media"] - x["map_min"]
    x["delta_lactato"] = x["lactato_max"] - x["lactato_media"]
    x["delta_creatinina"] = x["creatinina_max"] - x["creatinina_media"]
    x["delta_plaquetas"] = x["plaquetas_media"] - x["plaquetas_min"]

    # Para evitar divisiones por 0 o valores raros
    map_segura = x["map_min"].where(x["map_min"] > 0, np.nan)
    x["lactato_por_map"] = x["lactato_max"] / map_segura
    x["edad_por_lactato"] = x["anchor_age"] * x["lactato_max"]

    #indicadores de missing de las cols originales
    columnas_originales = df.drop(columns=fuera + [target]).columns

    for col in columnas_originales:
        if x[col].isnull().any():
            x[f"{col}_falta"] = x[col].isnull().astype(int)

    return x, y
"""""""""""

def preparar(df):
    target = "etiqueta_norad"
    fuera = ["stay_id", "subject_id", "hadm_id", "intime", "outtime", "rn","horas_uci", "inicio_noradrenalina", "horas_hasta_norad"]

    x = df.drop(columns=fuera + [target]).copy()
    y = df[target].copy()

    # indicadores de missing para todas las variables de x
    for col in x.columns:
        if x[col].isnull().any():
            x[f"{col}_falta"] = x[col].isnull().astype(int)

    return x, y

def dividir(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#-----------

def modelo_baseline(x_train, x_test, y_train, y_test):
    # Pipeline para que imputer y scaler se reajusten en cada fold
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()),
    ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced", C=5, solver="liblinear"))]) #saga y lbfgs NO dan mejor rendimiento

    # Validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc")
    print(f"AUC CV (media ± std): {auc.mean()} ± {auc.std()}")
    for i in auc:
        print(f"AUC OBTENIDO: {i}")

    # Entrena final sobre todo el train y evalúa en test
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred)
    print(f"AUC test final: {auc_test}")

#Nos quedamos con  C = 5 tras hacer pruebas porque a partir de este valor, la mejora no es signifivativa para nada y añadiríamos complejidad "absurdamente"

def main():
    df = cargar_datos()
    x, y = preparar(df)
    x_train, x_test, y_train, y_test = dividir(x, y)
    modelo_baseline(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()                                                                                                                                                                                                                                                                                 