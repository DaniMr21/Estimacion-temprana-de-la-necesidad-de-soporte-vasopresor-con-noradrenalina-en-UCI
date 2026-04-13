import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


def cargar_datos():
    contrasena = quote_plus("12345678")
    engine = create_engine(f"postgresql+psycopg2://postgres:{contrasena}@localhost:5432/MIMIC-IV")
    df = pd.read_sql("select * from public.dataset_final_v2", engine)
    return df


def preparar(df):
    target = "etiqueta_norad_6_24"
    fuera = ["stay_id", "subject_id", "hadm_id", "horas_hasta_norad", "contador_estancia_uci"]

    x = df.drop(columns=fuera + [target]).copy()
    y = df[target].copy()

    for col in x.columns:
        if x[col].isnull().any():
            x[f"{col}_falta"] = x[col].isnull().astype(int)

    return x, y


def dividir(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


def modelo_lightgbm(x_train, x_test, y_train, y_test):
    pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=3)),
        ("modelo", LGBMClassifier(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=42,
            verbosity=-1
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    print(f"AUC CV: {auc.mean():.4f} ± {auc.std():.4f}")
    for i, valor in enumerate(auc, start=1):
        print(f"  Fold {i}: {valor:.4f}")

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred)
    print(f"AUC test: {auc_test:.4f}")


def main():
    df = cargar_datos()
    x, y = preparar(df)
    x_train, x_test, y_train, y_test = dividir(x, y)
    modelo_lightgbm(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()