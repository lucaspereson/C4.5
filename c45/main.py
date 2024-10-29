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

c2 = C45(pathToCsv="D:\IA_C45\C4.5\data\data_cardiovascular_risk_TrainData.csv", gainRatio=False, infoGainThreshold=0.1)
c2.fetchDataCSV()
c2.preprocessData()
c2.generateTree()
c2.printTree()
c2.load_test_data("D:\IA_C45\C4.5\data\data_cardiovascular_risk_TestData.csv")
accuracy = c2.calculate_accuracy()
print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")
