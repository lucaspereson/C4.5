import math
import csv

class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, pathToData,pathToNames, pathToCsv):
		self.filePathToCsv = pathToCsv
		self.filePathToData = pathToData
		self.filePathToNames = pathToNames
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.tree = None
		self.indent = ""

	def fetchData(self):
		with open(self.filePathToNames, "r") as file:
			classes = file.readline()
			self.classes = [x.strip() for x in classes.split(",")]
			#add attributes
			for line in file:
				[attribute, values] = [x.strip() for x in line.split(":")]
				values = [x.strip() for x in values.split(",")]
				self.attrValues[attribute] = values
		self.numAttributes = len(self.attrValues.keys())
		self.attributes = list(self.attrValues.keys())
		print("attr: ", self.attrValues)
		print("---------------------------------")
		print("Attributes:",self.attributes)
		print("Classes:",self.classes)
		print("---------------------------------")
  
		with open(self.filePathToData, "r") as file:
			for line in file:
				row = [x.strip() for x in line.split(",")]
				if row != [] or row != [""]:
					self.data.append(row)
    
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
		for index,row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if(not self.isAttrDiscrete(self.attributes[attr_index])):
					self.data[index][attr_index] = float(self.data[index][attr_index])

	def printTree(self):
		self.printNode(self.tree)

	def printNode(self, node, indent=""):
		if not node.isLeaf:
			if node.threshold is None:
				#discrete
				for index,child in enumerate(node.children):
					if child.isLeaf:
						print(indent + node.label + " = " + self.attributes[index] + " : " + child.label)
					else:
						print(indent + node.label + " = " + self.attributes[index] + " : ")
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
		print("Generando árbol...")
		self.tree = self.recursiveGenerateTree(self.data, self.attributes)

	def recursiveGenerateTree(self, curData, curAttributes):
		self.indent += "-"
		

		if len(curData) == 0:
			#Fail
			print(f"{self.indent} Retorno nodo como etiqueta: Fail")
			return Node(True, "Fail", None)
		else:
			allSame = self.allSameClass(curData)
			if allSame is not False:
				#return a node with that class
				print(f"{self.indent} Retorno nodo como etiqueta: ",allSame)
				self.indent = self.indent[:-1]
				return Node(True, allSame, None)
			elif len(curAttributes) == 0:
				#return a node with the majority class
				majClass = self.getMajClass(curData)
				print(f"{self.indent} Retorno nodo con etiqueta de mayor clase: ",majClass)
				self.indent = self.indent[:-1]
				return Node(True, majClass, None)
			else:
				(best,best_threshold,splitted) = self.splitAttribute(curData, curAttributes)
				remainingAttributes = curAttributes[:]
				
				node = Node(False, best, best_threshold)
				if best != -1:
					remainingAttributes.remove(best)
					print(f"{self.indent} * Mejor atributo: ",best)
					print(f"{self.indent} Creando nodo en el árbol para: ",best)
					indexAttr = self.attributes.index(best)	
				node.children = []
				self.indent += "-"
				indexSplit = 0
				for subset in splitted:	
					if subset != []:
						if best_threshold is not None:
							if indexSplit == 0:
								print(f"{self.indent} Generando rama para subconjunto <={best_threshold} con {len(subset)} ejemplos.")
								indexSplit += 1
							else :
								print(f"{self.indent} Generando rama para subconjunto >{best_threshold} con {len(subset)} ejemplos.")
						else:
							print(f"{self.indent} Generando rama para subconjunto {subset[0][indexAttr]} con {len(subset)} ejemplos.")
					child = self.recursiveGenerateTree(subset, remainingAttributes)
					node.children.append(child)
				self.indent = self.indent[:-2]
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
		maxEnt = -1*float("inf")
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None
		for attribute in curAttributes:
			print(f"{self.indent} Evaluando Attr:",attribute, " con valores ",self.attrValues[attribute])
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
				e = self.gain(curData, subsets)
				if e > maxEnt:
					maxEnt = e
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
						e = self.gain(curData, [less, greater])
						if e >= maxEnt:
							splitted = [less, greater]
							maxEnt = e
							best_attribute = attribute
							best_threshold = threshold
		return (best_attribute,best_threshold,splitted)

	def gain(self, unionSet, subsets):
		#input : data and disjoint subsets of it
		#output : information gain
		S = len(unionSet)
		#calculate impurity before split
		impurityBeforeSplit = self.entropy(unionSet)
		#calculate impurity after split
		weights = [len(subset)/S for subset in subsets]
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
		#calculate total gain
		totalGain = impurityBeforeSplit - impurityAfterSplit
		return totalGain

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

class Node:
	def __init__(self,isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []


