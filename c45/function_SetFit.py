import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def set_fit(path, target_column, testSize=0.3):
    
    data = pd.read_csv(path)
    
    # Columna de referencia para estratificar
    stratify_column = data[target_column]

    # Configurar el muestreo estratificado
    split = StratifiedShuffleSplit(n_splits=1, test_size=testSize, random_state=42)

    # Dividir el dataset en 70% para entrenamiento y 30% para prueba
    for train_index, test_index in split.split(data, stratify_column):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

    # Guardar la muestra en un nuevo archivo CSV si lo deseas
    path = path.replace(".csv", "")
    pathTrain = path+"_TrainData.csv"
    pathTest = path+"_TestData.csv"
    train_data.to_csv(pathTrain, index=False)
    test_data.to_csv(pathTest, index=False)

    # Verificar la distribución en cada subconjunto
    print("Distribución en el conjunto de entrenamiento:")
    print(train_data['TenYearCHD'].value_counts(normalize=True))
    print("\nDistribución en el conjunto de prueba:")
    print(test_data['TenYearCHD'].value_counts(normalize=True))
    
    return pathTrain, pathTest


