import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

def cargar_datos():
    contrasena = quote_plus("12345678")
    engine = create_engine(f"postgresql+psycopg2://postgres:{contrasena}@localhost:5432/MIMIC-IV")
    df = pd.read_sql("select * from public.dataset_modelo_6h", engine)
    return df

def preparar(df):
    target = "etiqueta_norad"
    fuera = ["stay_id", "subject_id", "hadm_id", "intime", "outtime", "rn","horas_uci", "inicio_noradrenalina", "horas_hasta_norad"]

    x = df.drop(columns=fuera + [target]).copy()
    y = df[target].copy()

    #aqui tampoco meter v ariables derivadas porque no sube el aucroc y no hay casi diferencia

    # indicadores de missing para todas las variables de x
    for col in x.columns:
        if x[col].isnull().any():
            x[f"{col}_falta"] = x[col].isnull().astype(int)

    return x, y

def dividir(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#-------------------

def modelo_catboost(x_train, x_test, y_train, y_test):
    modelo = CatBoostClassifier(iterations=700, depth=2, learning_rate=0.03, loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(modelo, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    print(f"AUC CV: {auc.mean()} ± {auc.std()}")

    modelo.fit(x_train, y_train)
    y_pred = modelo.predict_proba(x_test)[:, 1]

    auc_test = roc_auc_score(y_test, y_pred)
    print(f"AUC test: {auc_test}")

def main():
    df = cargar_datos()
    x, y = preparar(df)
    x_train, x_test, y_train, y_test = dividir(x, y)
    modelo_catboost(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()  