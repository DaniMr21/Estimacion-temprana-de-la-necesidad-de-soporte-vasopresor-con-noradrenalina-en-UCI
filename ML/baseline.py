import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def cargar_datos():
    contrasena = quote_plus("12345678")
    engine = create_engine(f"postgresql+psycopg2://postgres:{contrasena}@localhost:5432/MIMIC-IV")
    df = pd.read_sql("select * from public.dataset_modelo_6h", engine)
    return df

def preparar(df):
    target = "etiqueta_norad"
    fuera = ["stay_id", "subject_id", "hadm_id","intime", "outtime", "rn", "horas_uci", "inicio_noradrenalina", "horas_hasta_norad"]
    x = df.drop(columns=fuera + [target]).copy()
    y = df[target].copy()

    #indicadores de missing, bien para ganar más info
    for col in x.columns:
        if x[col].isnull().any():
            x[f"{col}_falta"] = x[col].isnull().astype(int)

    return x, y


def dividir(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#-----------

def modelo_baseline(x_train, x_test, y_train, y_test):
    imputer = SimpleImputer(strategy="median")
    #imputer = SimpleImputer(strategy="mean") baja un poco el AUC

    x_train_imp = imputer.fit_transform(x_train)
    x_test_imp = imputer.transform(x_test)

    scaler = StandardScaler() #esclar para la RegLog
    x_train_imp = scaler.fit_transform(x_train_imp)
    x_test_imp = scaler.transform(x_test_imp)

    modelo = LogisticRegression(max_iter=3000, class_weight="balanced", C=5, solver="liblinear")

    modelo.fit(x_train_imp, y_train)
    y_pred = modelo.predict_proba(x_test_imp)[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC: {auc}")

#Nos quedamos con  C = 5 tras hacer pruebas porque a partir de este valor, la mejora no es signifivativa para nada y añadiríamos complejidad "absurdamente"

def main():
    df = cargar_datos()
    x, y = preparar(df)
    x_train, x_test, y_train, y_test = dividir(x, y)

    modelo_baseline(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()                                                                                                                                                                                                                                                                                 