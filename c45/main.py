#!/usr/bin/env python
import pdb
from c45 import C45
"""
c1 = C45("D:\IA_C45\C4.5\data\iris\iris.data", "D:\IA_C45\C4.5\data\iris\iris.names", None)
c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()


c2 = C45("D:\IA_C45\C4.5\data\prob_cardiacos\pc.data", "D:\IA_C45\C4.5\data\prob_cardiacos\pc.names", None)
c2.fetchData()
c2.preprocessData()
c2.generateTree()
c2.printTree()
"""

#c2 = C45(None, None, "D:\IA_C45\C4.5\data\data_cardiovascular_risk.csv")
c2 = C45(None, None, "D:\IA_C45\C4.5\data\data_example.csv")
c2.fetchDataCSV()
c2.preprocessData()
c2.generateTree()
#c2.printTree()