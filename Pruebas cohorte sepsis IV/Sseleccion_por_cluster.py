import os
import pandas as pd

CARPETA = r"C:\Users\danie\TFG\Pruebas cohorte sepsis IV"
CARPETA_TABLAS = os.path.join(CARPETA, "tablas")

RUTA_CLUSTERS = os.path.join(CARPETA_TABLAS, "clusters_variables_v4.csv")
RUTA_RF = os.path.join(CARPETA, "importancia_variables_rf_v4.txt")
RUTA_SALIDA = os.path.join(CARPETA_TABLAS, "seleccion_variables_por_cluster_v4.csv")

clusters = pd.read_csv(RUTA_CLUSTERS)

rf = pd.read_csv(
    RUTA_RF,
    sep=r"\s+",
    skiprows=3,
    names=["variable", "importancia"],
    engine="python"
)

df = clusters.merge(rf, on="variable", how="left")

seleccion = (
    df.sort_values(["cluster", "importancia"], ascending=[True, False])
      .groupby("cluster")
      .first()
      .reset_index()
)

seleccion = seleccion[["cluster", "variable", "importancia"]]

seleccion.to_csv(RUTA_SALIDA, index=False)

print("\nSELECCIÓN AUTOMÁTICA POR CLUSTER + RF")
print("=" * 60)
print(seleccion.to_string(index=False))
print(f"\nGuardado en: {RUTA_SALIDA}")

print("\nLista para copiar en Python:")
print("[")
for v in seleccion["variable"]:
    print(f"    '{v}',")
print("]")