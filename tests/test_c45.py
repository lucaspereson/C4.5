#!/usr/bin/env python
import unittest
from c45.c45 import C45

class testC45Methods(unittest.TestCase):
	c1 = C45(None, None, "D:\IA_C45\C4.5\data\data_cardiovascular_risk_TrainData.csv")

	def testFoo(self):
		self.assertEqual(True, True)


def main():
	unittest.main()

if __name__ == '__main__':
	main()

