import math
import csv
import time

class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, pathToCsv, gainRatio=False, infoGainThreshold=0.001):
		self.filePathToCsv = pathToCsv
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.attributesToRemove = []
		self.tree = None
		self.indent = ""
		self.timeStart = 0
		self.timeEnd = 0
		self.infoGainThreshold = infoGainThreshold
		self.gainRatio = gainRatio
    
	def fetchDataCSV(self):
		with open(self.filePathToCsv, "r") as file:
			reader = csv.reader(file)
			# Leer la primera fila (nombres de atributos)
			self.attributes = next(reader)
			self.attributes.pop()
			self.numAttributes = len(self.attributes)
			print("---------------------------------")
			print("Attributes:", self.attributes)
			# Leer el resto de los datos
			for row in reader:
				if all(element.strip() for element in row):
					self.data.append(row)
			
			# Inferir las clases (última columna)
			self.classes = list(set([row[-1] for row in self.data]))
			print("Classes:", self.classes)

			# Inferir los valores posibles de los atributos
			self.inferAttributeValues()
			print("Attr Values:", self.attrValues)
			print("---------------------------------")
   	
	def inferAttributeValues(self):
		for i in range(self.numAttributes):
			column_values = [row[i] for row in self.data]
			try:
				# Intentar convertir todos los valores a float para determinar si es continuo
				list(map(float, column_values))
				self.attrValues[self.attributes[i]] = ["continuous"]
			except ValueError:
				# Si falla, se considera discreto y se almacenan sus valores únicos
				self.attrValues[self.attributes[i]] = list(set(column_values))

	def preprocessData(self):
		# Verificar si un atributo tiene valores únicos en cada registro
		for attr_index in range(self.numAttributes):
			unique_values = set(row[attr_index] for row in self.data)
			if len(unique_values) == len(self.data):
				self.attributesToRemove.append(attr_index)

		# Eliminar los atributos únicos de cada registro y actualizar la lista de atributos
		for index in sorted(self.attributesToRemove, reverse=True):
			for row in self.data:
				del row[index]
			print(f"Eliminando atributo único: {self.attributes[index]}")
			del self.attrValues[self.attributes[index]]
			del self.attributes[index]
			self.numAttributes -= 1
		# Convertir valores a float para atributos no discretos
		for index, row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if not self.isAttrDiscrete(self.attributes[attr_index]):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	def printTree(self):
		self.printNode(self.tree)

	def printNode(self, node, indent=""):
		if not node.isLeaf:
			if node.threshold is None:
				#discrete
				for index,child in enumerate(node.children):
					if child.isLeaf:
						print(indent + node.label + " = " + self.attrValues[node.label][index] + " : " + child.label)
					else:
						print(indent + node.label + " = " + self.attrValues[node.label][index] + " : ")
						self.printNode(child, indent + "	")
			else:
				#numerical
				leftChild = node.children[0]
				rightChild = node.children[1]
				if leftChild.isLeaf:
					print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
				else:
					print(indent + node.label + " <= " + str(node.threshold)+" : ")
					self.printNode(leftChild, indent + "	")

				if rightChild.isLeaf:
					print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
				else:
					print(indent + node.label + " > " + str(node.threshold) + " : ")
					self.printNode(rightChild , indent + "	")

	def generateTree(self):
		self.timeStart = time.time()
		print("Generando árbol...")
		self.tree = self.recursiveGenerateTree(self.data, self.attributes)
		self.timeEnd = time.time()
		print("Árbol generado en ",self.timeEnd - self.timeStart," segundos.")

	def recursiveGenerateTree(self, curData, curAttributes):
		if len(curData) == 0:
			#Fail
			return Node(True, "Fail", None)
		else:
			allSame = self.allSameClass(curData)
			if allSame is not False:
				#return a node with that class
				return Node(True, allSame, None)
			elif len(curAttributes) == 0:
				#return a node with the majority class
				majClass = self.getMajClass(curData)
				return Node(True, majClass, None)
			else:
				(best,best_threshold,splitted) = self.splitAttribute(curData, curAttributes)
				remainingAttributes = curAttributes[:]
				if best != -1: # Hay un atributo para particionar
					node = Node(False, best, best_threshold)
					node.children = []
					if best_threshold is None and best != -1: #atributo discreto
						remainingAttributes.remove(best)
					for subset in splitted:	
						child = self.recursiveGenerateTree(subset, remainingAttributes)
						node.children.append(child)
				else:
					#return a node with the majority class
					majClass = self.getMajClass(curData)
					self.indent = self.indent[:-1]
					return Node(True, majClass, None)
				return node

	def getMajClass(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]

	def allSameClass(self, data):
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	def isAttrDiscrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
			return False
		else:
			return True

	def splitAttribute(self, curData, curAttributes):
		splitted = []
		#minGan = -1*float("inf")
		minGan = self.infoGainThreshold
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None
		for attribute in curAttributes:
			#print(f"{self.indent} Evaluando Attr:",attribute, " con valores ",self.attrValues[attribute])
			indexOfAttribute = self.attributes.index(attribute)
			if self.isAttrDiscrete(attribute):
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				valuesForAttribute = self.attrValues[attribute]
				subsets = [[] for a in valuesForAttribute]
				for row in curData:
					for index in range(len(valuesForAttribute)):
						if row[indexOfAttribute] == valuesForAttribute[index]:
							subsets[index].append(row)
							break
				g = self.gain(curData, subsets)
				if g > minGan:
					minGan = g
					splitted = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				curData.sort(key = lambda x: x[indexOfAttribute])
				for j in range(0, len(curData) - 1):
					if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
						threshold = (curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
						less = []
						greater = []
						for row in curData:
							if(row[indexOfAttribute] > threshold):
								greater.append(row)
							else:
								less.append(row)
						g = self.gain(curData, [less, greater])
						if g >= minGan:
							splitted = [less, greater]
							minGan = g
							best_attribute = attribute
							best_threshold = threshold
				if 0 >= minGan:
					splitted = [curData, []]
					minGan = 0
					best_attribute = attribute
					best_threshold = curData[0][indexOfAttribute]
		return (best_attribute,best_threshold,splitted)

	def gain(self, unionSet, subsets):
		# Entrada: conjunto de datos y subconjuntos disjuntos
    	# Salida: ganancia de información y tasa de ganancia
		S = len(unionSet)
    	
     	# Calcular la impureza antes de la división
		impurityBeforeSplit = self.entropy(unionSet)
    	
     	# Calcular la impureza después de la división
		weights = [len(subset)/S for subset in subsets]
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
		
  		# Calcular la ganancia de información total
		totalGain = impurityBeforeSplit - impurityAfterSplit

		if self.gainRatio == False:
			return totalGain
		
  		# Calcular la entropía de partición (split info)
		splitInfo = -sum(weight * math.log2(weight) for weight in weights if weight > 0)

		# Calcular la tasa de ganancia
		gainRatio = totalGain / splitInfo if splitInfo != 0 else 0

		return gainRatio

	def entropy(self, dataSet):
		S = len(dataSet)
		if S == 0:
			return 0
		num_classes = [0 for i in self.classes]
		for row in dataSet:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] += 1
		num_classes = [x/S for x in num_classes]
		ent = 0
		for num in num_classes:
			ent += num*self.log(num)
		return ent*-1


	def log(self, x):
		if x == 0:
			return 0
		else:
			return math.log(x,2)

	def load_test_data(self, path_to_test_csv):
		self.test_data = []
		with open(path_to_test_csv, "r") as file:
			reader = csv.reader(file)
			next(reader)  # Saltar la fila de cabecera si existe
			for row in reader:
				if all(element.strip() for element in row):
					self.test_data.append(row)
     
		for index in sorted(self.attributesToRemove, reverse=True):
			for row in self.test_data:
				del row[index]
    
	def predict(self, instance, node=None):
		"""Predice la clase de una instancia recorriendo el árbol."""
		if node is None:
			node = self.tree
		if node.isLeaf:
			#print(f" - Predicción para: {instance}, clase: {node.label}")
			return node.label
		else:
			#print(f"Evaluando instancia: {instance}, para el nodo: {node.label}")
			attr_index = self.attributes.index(node.label)
			attr_value = instance[attr_index]
			if node.threshold is None:
				# Atributo discreto
				for i, child in enumerate(node.children):
					if self.attrValues[node.label][i] == attr_value:
						#print(f"Nodo encontrado: {node.label} = {attr_value}")
						#print(f"Siguiente nodo: {child.label}")
						return self.predict(instance, child)
			else:
				# Atributo continuo
				if float(attr_value) <= node.threshold:
					#print(f"Nodo encontrado: {node.label} <= {node.threshold}")
					return self.predict(instance, node.children[0])
				else:
					#print(f"Nodo encontrado: {node.label} > {node.threshold}")
					return self.predict(instance, node.children[1])

	def calculate_accuracy(self):
		"""Calcula la precisión del modelo en el conjunto de prueba."""
		correct_predictions = 0
		for instance in self.test_data:
			actual_class = instance[-1]  # Último valor es la clase
			predicted_class = self.predict(instance)
			if actual_class == predicted_class:
				correct_predictions += 1
		accuracy = correct_predictions / len(self.test_data)
		return accuracy

class Node:
	def __init__(self,isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []


