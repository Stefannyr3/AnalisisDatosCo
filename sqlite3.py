import sqlite3

#PUNTO DE LA BASE DE DATOS
#Conexión a la base de datos
conn = sqlite3.connect("dbcovid.db")
c = conn.cursor()
c.execute("""CREATE TABLE datos(
USMER REAL PRIMARY KEY,
MEDICAL_UNIT REAL,
SEX REAL,
PATIENT_TYPE REAL,
DATE_DIED REAL,
INTUBED REAL,
PNEUMONIA REAL,
AGE REAL,
PREGNANT REAL,
DIABETES REAL,
COPD REAL,
ASTHMA REAL,
INMSUPR REAL,
HIPERTENSION REAL,
OTHER_DISEASE REAL,
CARDIOVASCULAR REAL,
OBESITY REAL,
RENAL_CHRONIC REAL,
TOBACCO REAL,
CLASIFFICATION_FINAL REAL,
ICU REAL)""")

# Verificar la integridad de las claves únicas
datos = "datos"
USMER = "USMER"
unique_key_check = conn.execute(f"SELECT COUNT(*) FROM {datos} WHERE {USMER} NOT IN (SELECT {USMER} FROM {datos} GROUP BY {USMER} HAVING COUNT(*) = 1)").fetchone()
print("Verificación de claves únicas:", unique_key_check[0] == 0)

# Cerrar la conexión a la base de datos
conn.close()

