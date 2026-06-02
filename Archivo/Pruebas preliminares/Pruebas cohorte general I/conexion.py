import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

contrasena = quote_plus("12345678")

motor = create_engine(f"postgresql+psycopg2://postgres:{contrasena}@localhost:5432/MIMIC-IV")

consulta = "select * from public.dataset_modelo_6h"
resultado = pd.read_sql(consulta, motor)

print(resultado.shape)
print(resultado.columns.tolist())
print(resultado.head())
print(resultado.isnull().sum().sort_values(ascending=False))