import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Cargar el archivo CSV
dataset = pd.read_csv("D:\IA_C45\C4.5\data\data_cardiovascular_risk.csv")

# Columna de referencia para estratificar (por ejemplo, la columna 'class')
# Asegúrate de cambiar 'class' al nombre de tu columna de interés
stratify_column = dataset["TenYearCHD"]

# Configurar el muestreo estratificado
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Dividir el dataset en 80% para entrenamiento y 20% para prueba
for train_index, test_index in split.split(dataset, stratify_column):
    train_data = dataset.iloc[train_index]
    test_data = dataset.iloc[test_index]

# Guardar la muestra en un nuevo archivo CSV si lo deseas
train_data.to_csv("D:\IA_C45\C4.5\data\data_cardiovascular_risk_TrainData.csv", index=False)
test_data.to_csv("D:\IA_C45\C4.5\data\data_cardiovascular_risk_TestData.csv", index=False)

# Verificar la distribución en cada subconjunto
print("Distribución en el conjunto de entrenamiento:")
print(train_data['TenYearCHD'].value_counts(normalize=True))
print("\nDistribución en el conjunto de prueba:")
print(test_data['TenYearCHD'].value_counts(normalize=True))
