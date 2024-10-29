#!/usr/bin/env python
from c45 import C45
from function_SetFit import set_fit

pathTrain, pathTest = set_fit(path="D:\IA_C45\C4.5\data\data_cardiovascular_imputado.csv", target_column="TenYearCHD", testSize=0.3)

c2 = C45(pathToCsv=pathTrain, gainRatio=False, infoGainThreshold=0.001)
c2.fetchDataCSV()
c2.preprocessData()
c2.generateTree()
c2.printTree()
c2.load_test_data(path_to_test_csv=pathTest)
total_instances_0, total_instances_1, accuracy, accuracy_0, accuracy_1 = c2.calculate_accuracy()
print(f"Total de instancias en el conjunto de prueba: {total_instances_0+total_instances_1}")
print(f"Total de instancias de la clase 0 en el conjunto de prueba: {total_instances_0}")
print(f"Total de instancias de la clase 1 en el conjunto de prueba: {total_instances_1}")
print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")
print(f"Precisión del modelo en el conjunto de prueba para la clase 0: {accuracy_0 * 100:.2f}%")
print(f"Precisión del modelo en el conjunto de prueba para la clase 1: {accuracy_1 * 100:.2f}%")
c2.confusion_matrix_c45()
