#!/usr/bin/env python
from c45 import C45
from function_SetFit import set_fit
from function_SortedCSV import sortedCsv

sortedCsv(input_csv="D:\IA_C45\C4.5\data\data_cardiovascular_imputado.csv", output_csv="D:\IA_C45\C4.5\data\data_cardiovascular_imputado_sorted.csv")
pathTrain, pathTest = set_fit(path="D:\IA_C45\C4.5\data\data_cardiovascular_imputado_sorted.csv", target_column="TenYearCHD", testSize=0.3)
try:
    c2 = C45(pathToCsv=pathTrain, gainRatio=False, infoGainThreshold=0.003)
    c2.fetchDataCSV()
    c2.preprocessData()
    c2.generateTree()
    c2.printTree()
    c2.loadDataToPredict(path_to_test_csv=pathTest)
    total_instances_0, total_instances_1, accuracy, accuracy_0, accuracy_1, recall_0, recall_1 = c2.makePredictions()
    print(f"Total de instancias en el conjunto de prueba: {total_instances_0+total_instances_1}")
    print(f"Total de instancias de la clase 0 en el conjunto de prueba: {total_instances_0}")
    print(f"Total de instancias de la clase 1 en el conjunto de prueba: {total_instances_1}")
    print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")
    print(f"Precisión del modelo en el conjunto de prueba para la clase 0 (Sensibilidad): {accuracy_0 * 100:.2f}%")
    print(f"Precisión del modelo en el conjunto de prueba para la clase 1 (Sensibilidad): {accuracy_1 * 100:.2f}%")
    print(f"Recall de la clase 0: {recall_0}")
    print(f"Recall de la clase 1: {recall_1}")
    c2.confusionMatrixC45()
except Exception as e:
    print(e)