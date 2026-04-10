import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

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

def modelo_xg(x_train, x_test, y_train, y_test):
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean")), #con mean ahora sí que mejora algo respecto a median
        ("modelo", XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.7, colsample_bytree=1, scale_pos_weight=0.7, objective="binary:logistic",eval_metric="auc",random_state=42))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc")

    print(f"AUC CV (media ± std): {auc.mean()} ± {auc.std()}")
    for i, valor in enumerate(auc, start=1):
        print(f"fold {i}: {valor}")

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred)

    print(f"AUC test final: {auc_test}")

def main():
    df = cargar_datos()
    x, y = preparar(df)
    x_train, x_test, y_train, y_test = dividir(x, y)
    modelo_xg(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()    