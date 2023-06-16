import pandas as pd
import numpy as np
#convertir .csv a un dataframe
dataframe = pd.read_csv('CovidData.csv',delimiter=';')
"""tomar los datos de la columna date_died con los datos 9999-99-99 el cual
es un dato que indica que no hay fecha de fallecimiento y se transformara a nan 
variable nula python"""
dataframe['DATE_DIED'] = dataframe['DATE_DIED'].replace('9999-99-99', np.nan)
"""tomar los datos que sean 1 o 2 y se transformara a True o False para mirar que datos 
son verdaderos o falsos y los datos que no tengan este valor se remplazara por nan variable
nula en python"""
replacement_dict = {1: True, 2: False}
dataframe['INTUBED'] = dataframe['INTUBED'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['INTUBED'].isin([True, False]), 'INTUBED'] = np.nan
"""Se revisa si todos los datos de todas las columnas son numericos y se procede a cambiar 
los datos que tengan valores de 99 a nan variable nula en python"""
is_numeric = pd.to_numeric(dataframe['AGE'], errors='coerce').notnull().all()
if is_numeric:
    print("All values in the column are numbers.")
else:
    print("There are non-numeric values in the column.")
"""Se toman los datos de la columna age y se cambian por nan los datos que sean mayores a 100"""
dataframe.loc[dataframe['AGE'] > 100, 'AGE'] = np.nan
has_numbers_greater_than_100 = (dataframe['AGE'] > 100).any()

if has_numbers_greater_than_100:
    print("There are numbers greater than 100 in the column.")
else:
    print("There are no numbers greater than 100 in the column.")

dataframe['PREGNANT'] = dataframe['PREGNANT'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['PREGNANT'].isin([True, False]), 'PREGNANT'] = np.nan

dataframe['DIABETES'] = dataframe['DIABETES'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['DIABETES'].isin([True, False]), 'DIABETES'] = np.nan

dataframe['COPD'] = dataframe['COPD'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['COPD'].isin([True, False]), 'COPD'] = np.nan

dataframe['ASTHMA'] = dataframe['ASTHMA'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['ASTHMA'].isin([True, False]), 'ASTHMA'] = np.nan

dataframe['INMSUPR'] = dataframe['INMSUPR'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['INMSUPR'].isin([True, False]), 'INMSUPR'] = np.nan

dataframe['HIPERTENSION'] = dataframe['HIPERTENSION'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['HIPERTENSION'].isin([True, False]), 'HIPERTENSION'] = np.nan

dataframe['OTHER_DISEASE'] = dataframe['OTHER_DISEASE'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['OTHER_DISEASE'].isin([True, False]), 'OTHER_DISEASE'] = np.nan

dataframe['CARDIOVASCULAR'] = dataframe['CARDIOVASCULAR'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['CARDIOVASCULAR'].isin([True, False]), 'CARDIOVASCULAR'] = np.nan

dataframe['OBESITY'] = dataframe['OBESITY'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['OBESITY'].isin([True, False]), 'OBESITY'] = np.nan

dataframe['RENAL_CHRONIC'] = dataframe['RENAL_CHRONIC'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['RENAL_CHRONIC'].isin([True, False]), 'RENAL_CHRONIC'] = np.nan

dataframe['TOBACCO'] = dataframe['TOBACCO'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['TOBACCO'].isin([True, False]), 'TOBACCO'] = np.nan

dataframe['ICU'] = dataframe['ICU'].replace(replacement_dict, regex=False)
dataframe.loc[~dataframe['ICU'].isin([True, False]), 'ICU'] = np.nan

columna = dataframe.loc[:10, 'ICU']
print(columna)
