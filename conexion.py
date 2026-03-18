import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

contrasena = quote_plus("12345678")

motor = create_engine(f'postgresql+psycopg2://postgres:{contrasena}@localhost:5432/MIMIC-IV')

consulta = "SELECT COUNT(*) AS total_pacientes FROM mimiciv_hosp.admissions"
resultado = pd.read_sql(consulta, motor)
print(resultado)