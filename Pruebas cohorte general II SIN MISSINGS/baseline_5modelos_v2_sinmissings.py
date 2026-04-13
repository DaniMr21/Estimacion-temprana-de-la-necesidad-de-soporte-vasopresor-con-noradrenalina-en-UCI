import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def cargar_datos():
    contrasena = quote_plus("12345678")
    engine = create_engine(f"postgresql+psycopg2://postgres:{contrasena}@localhost:5432/MIMIC-IV")
    df = pd.read_sql("select * from public.dataset_final_v2_sinmissing", engine)
    return df


def preparar(df):
    target = "etiqueta_norad_6_24"
    fuera = ["stay_id", "subject_id", "hadm_id", "horas_hasta_norad", "contador_estancia_uci"]

    x = df.drop(columns=fuera + [target]).copy()
    y = df[target].copy()
    return x, y


def dividir(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


def evaluar(nombre, pipeline, x_train, x_test, y_train, y_test, es_catboost=False):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc",
                          n_jobs=1 if es_catboost else -1)
    print(f"\n{'='*40}")
    print(f"MODELO: {nombre}")
    print(f"AUC CV: {auc.mean():.4f} ± {auc.std():.4f}")
    for i, v in enumerate(auc, start=1):
        print(f"  Fold {i}: {v:.4f}")

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_pred)
    print(f"AUC test: {auc_test:.4f}")


def main():
    df = cargar_datos()
    print(f"Filas: {len(df)} | Positivos: {df['etiqueta_norad_6_24'].sum()}")

    x, y = preparar(df)
    x_train, x_test, y_train, y_test = dividir(x, y)

    # 1. Regresión Logística
    evaluar("Regresión Logística",
        Pipeline([
            ("scaler", RobustScaler()),
            ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced", C=5, solver="liblinear"))
        ]), x_train, x_test, y_train, y_test)

    # 2. Random Forest
    evaluar("Random Forest",
        Pipeline([
            ("modelo", RandomForestClassifier(n_estimators=600, max_depth=None,
                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, random_state=42))
        ]), x_train, x_test, y_train, y_test)

    # 3. XGBoost
    evaluar("XGBoost",
        Pipeline([
            ("modelo", XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03,
                subsample=0.7, colsample_bytree=1, scale_pos_weight=0.7,
                objective="binary:logistic", eval_metric="auc", random_state=42))
        ]), x_train, x_test, y_train, y_test)

    # 4. LightGBM
    evaluar("LightGBM",
        Pipeline([
            ("modelo", LGBMClassifier(n_estimators=300, max_depth=2, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.6, random_state=42, verbosity=-1))
        ]), x_train, x_test, y_train, y_test)

    # 5. CatBoost
    evaluar("CatBoost",
        CatBoostClassifier(iterations=700, depth=2, learning_rate=0.03,
            loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=0),
        x_train, x_test, y_train, y_test, es_catboost=True)


if __name__ == "__main__":
    main()