#!/usr/bin/env python
import pdb
from c45 import C45
"""
c2 = C45("D:\IA_C45\C4.5\data\prob_cardiacos\pc.csv")
c2.fetchDataCSV()
c2.preprocessData()
c2.generateTree()
c2.printTree()
"""
"""
c2 = C45("D:\IA_C45\C4.5\data\data_example.csv")
c2.fetchDataCSV()
c2.preprocessData()
c2.generateTree()
c2.printTree()
c2.load_test_data("D:\IA_C45\C4.5\data\data_example.csv")
accuracy = c2.calculate_accuracy()
print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")
"""

c2 = C45(pathToCsv="D:\IA_C45\C4.5\data\data_cardiovascular_risk_TrainData.csv", gainRatio=False, infoGainThreshold=0.001)
c2.fetchDataCSV()
c2.preprocessData()
c2.generateTree()
c2.printTree()
c2.load_test_data("D:\IA_C45\C4.5\data\data_cardiovascular_risk_TestData.csv")
total_instances_0, total_instances_1, accuracy, accuracy_0, accuracy_1 = c2.calculate_accuracy()
print(f"Total de instancias en el conjunto de prueba: {total_instances_0+total_instances_1}")
print(f"Total de instancias de la clase 0 en el conjunto de prueba: {total_instances_0}")
print(f"Total de instancias de la clase 1 en el conjunto de prueba: {total_instances_1}")
print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")
print(f"Precisión del modelo en el conjunto de prueba para la clase 0: {accuracy_0 * 100:.2f}%")
print(f"Precisión del modelo en el conjunto de prueba para la clase 1: {accuracy_1 * 100:.2f}%")
