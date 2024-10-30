import math
import csv
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class C45:
	def __init__(self, pathToCsv, gainRatio=False, infoGainThreshold=0.001):
		self.filePathToCsv = pathToCsv
		self.data = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.attributesToRemove = []
		self.tree = None
		self.timeStart = 0
		self.timeTotalFit = 0
		self.timeTotalPredict = 0
		self.infoGainThreshold = infoGainThreshold
		self.gainRatio = gainRatio
		self.y_test = []
		self.y_pred = []
    
	def fetchDataCSV(self):
		with open(self.filePathToCsv, "r") as file:
			reader = csv.reader(file)
			# Lee la primera fila (nombres de atributos)
			self.attributes = next(reader)
			self.attributes.pop()
			self.numAttributes = len(self.attributes)
			print("---------------------------------")
			print("Atributos:", self.attributes)
			# Lee el resto de los datos
			for row in reader:
				if all(element.strip() for element in row):
					self.data.append(row)
			
			# Inferir las clases (última columna)
			self.classes = list(set([row[-1] for row in self.data]))
			print("Clases:", self.classes)

			# Inferir los valores posibles de los atributos
			self.inferAttributeValues()
			print("Valores de atributos:", self.attrValues)
			print("---------------------------------")
   	
	def inferAttributeValues(self):
		for i in range(self.numAttributes):
			column_values = [row[i] for row in self.data]
			try:
				# Se intenta convertir todos los valores a float para determinar si es continuo
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
				# Cuando un atributo es discreto, el threshold del nodo es None
				for index,child in enumerate(node.children):
					if child.isLeaf:
						print(indent + node.label + " = " + self.attrValues[node.label][index] + " : " + child.label)
					else:
						print(indent + node.label + " = " + self.attrValues[node.label][index] + " : ")
						self.printNode(child, indent + "	")
			else:
				# Cuando un atributo es continuo, el threshold del nodo es un valor numérico que es el punto medio entre dos valores adyacentes
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
		else:
			# Esto sucede cuando el arbol es un único nodo hoja
			print(indent + "Nodo con clase: " + node.label)
   
	def generateTree(self):
		self.timeStart = time.time()
		print("Generando árbol...")
		# Se llama a la función recursiva para generar el árbol
		self.tree = self.recursiveGenerateTree(self.data, self.attributes)
		self.timeTotalFit = time.time() - self.timeStart
		print("Árbol generado en ",self.timeTotalFit," segundos.")

	def recursiveGenerateTree(self, curData, curAttributes):
		if len(curData) == 0:
			#Fail
			return Node(True, "Fail", None)
		else:
			allSame = self.allSameClass(curData)
			if allSame is not False:
				# Retorna un nodo con la clase a la que pertenecen todos los datos
				return Node(True, allSame, None)
			elif len(curAttributes) == 0:
				# Retorna un nodo con la clase mayoritaria
				majClass = self.getMajClass(curData)
				return Node(True, majClass, None)
			else:
				(best,best_threshold,splitted) = self.splitAttribute(curData, curAttributes)
				remainingAttributes = curAttributes[:]
				if best != -1: # Existe un atributo para particionar
					node = Node(False, best, best_threshold)
					node.children = []
					if best_threshold is None and best != -1: 
         				# El atributo es discreto, por lo que se lo remueve de la lista de atributos restantes de esa rama
						remainingAttributes.remove(best)
					# Se llama recursivamente a la función para cada subconjunto del atributo (son 2 si es continuo)
					for subset in splitted:	
						child = self.recursiveGenerateTree(subset, remainingAttributes)
						node.children.append(child)
					return node
				else: # No se puede particionar por ningun atributo de los restantes (la ganancia es menor al umbral)
					# Retorna un nodo con la clase mayoritaria
					majClass = self.getMajClass(curData)
					return Node(True, majClass, None)

	def getMajClass(self, curData):
		# Retorna la clase mayoritaria de un conjunto de datos
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]

	def allSameClass(self, data):
		# Retorna la clase si todos los datos pertenecen a la misma clase, False en caso contrario
		for row in data:
			if row[-1] != data[0][-1]:
				return False
		return data[0][-1]

	def isAttrDiscrete(self, attribute):
		# Retorna True si el atributo es discreto, False si es continuo
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
			return False
		else:
			return True

	def splitAttribute(self, curData, curAttributes):
		splitted = []
		minGan = self.infoGainThreshold
		best_attribute = -1
		# Se inicializa el mejor umbral en None, el cual permanecerá así si el atributo es discreto
		best_threshold = None
		for attribute in curAttributes:
			indexOfAttribute = self.attributes.index(attribute)
			if self.isAttrDiscrete(attribute):
				# Se divide el conjunto de datos en n-subconjuntos, donde n es la cantidad de valores diferentes 
    			# del atributo i. Se elige el atributo con la ganancia máxima
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
				# Se ordenan los datos según la columna. Luego se prueban todos los pares adyacentes posibles.
    			# Se divide el conjunto de datos en dos subconjuntos, donde el umbral es el punto medio entre 
    			# dos valores adyacentes. Se elige el atributo con la ganancia máxima
				curData.sort(key = lambda x: x[indexOfAttribute])
				for j in range(0, len(curData) - 1):
					if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]: 
         				# Si no son iguales se puede dividir en ese punto medio entre los dos valores
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
				# Esto se realiza cuando todos los valores del atributo continuo son iguales y no encuentra una particion para el atributo
				if 0 >= minGan:
					splitted = [curData, []]
					minGan = 0
					best_attribute = attribute
					best_threshold = curData[0][indexOfAttribute]
		return (best_attribute,best_threshold,splitted)

	def gain(self, unionSet, subsets):
		# Entrada: conjunto de datos y subconjuntos disjuntos
    	# Salida: ganancia de información o tasa de ganancia, de acuerdo a la configuración
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

		# Si se está utilizando la tasa de ganancia, se calcula el split info	
  		# Calcular la entropía de partición (split info)
		splitInfo = -sum(weight * math.log2(weight) for weight in weights if weight > 0)
		# Calcular la tasa de ganancia
		gainRatio = totalGain / splitInfo if splitInfo != 0 else 0
		return gainRatio

	def entropy(self, dataSet):
		# Calcular la entropía de un conjunto de datos
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
		# Calcular el logaritmo en base 2 de x
		if x == 0:
			return 0
		else:
			return math.log(x,2)

	def load_test_data(self, path_to_test_csv):
		# Cargar los datos de prueba de un csv
		self.test_data = []
		with open(path_to_test_csv, "r") as file:
			reader = csv.reader(file)
			next(reader)  # Saltar la fila de cabecera si existe ya que solo se necesita la data
			for row in reader:
				if all(element.strip() for element in row):
					self.test_data.append(row)

		# Se eliminan los atributos únicos de cada registro y se actualiza la lista de atributos, 
  		# para que al hacer la predicción recorriendo el árbol no haya errores
		for index in sorted(self.attributesToRemove, reverse=True):
			for row in self.test_data:
				del row[index]
    
	def predict(self, instance, node=None):
		# Predice la clase de una instancia recorriendo el árbol
		if node is None: # Comienza desde la raíz, esto solo se llama la primera vez
			node = self.tree
		if node.isLeaf: # Si es un nodo hoja, retorna la clase
			return node.label
		else: # Si no es un nodo hoja, sigue recorriendo el árbol
			attr_index = self.attributes.index(node.label)
			attr_value = instance[attr_index]
			if node.threshold is None:
				# Atributo discreto
				for i, child in enumerate(node.children):
					if self.attrValues[node.label][i] == attr_value:
						return self.predict(instance, child)
			else:
				# Atributo continuo
				if float(attr_value) <= node.threshold:
					return self.predict(instance, node.children[0])
				else:
					return self.predict(instance, node.children[1])

	def calculate_accuracy(self):
		# Calcula la precisión del modelo en el conjunto de prueba
		self.timeStart = time.time()
		correct_predictions_0 = 0
		correct_predictions_1 = 0
		predictions_0 = 0
		predictions_1 = 0
		total_instances_0 = 0
		total_instances_1 = 0
		self.y_test = []
		self.y_pred = []
		for instance in self.test_data:
			actual_class = instance[-1]  # El último valor es la clase
			self.y_test.append(actual_class)
			predicted_class = self.predict(instance)
			self.y_pred.append(predicted_class)
			if predicted_class == "0": # Cantidad de predicciones de 0
				predictions_0 += 1
			else:                   # Cantidad de predicciones de 1
				predictions_1 += 1
			if actual_class == "0":     # Cantidad de instancias de 0
				total_instances_0 += 1
			else:                   # Cantidad de instancias de 1
				total_instances_1 += 1
			if actual_class == predicted_class:
				if actual_class == "0": # Predicciones correctas de 0
					correct_predictions_0 += 1
				else:                   # Predicciones correctas de 1
					correct_predictions_1 += 1
		accuracy = (correct_predictions_0+correct_predictions_1) / (total_instances_0+total_instances_1)
		if predictions_0 == 0: 
			accuracy_0 = 0
		else:
			accuracy_0 = correct_predictions_0 / predictions_0
		if predictions_1 == 0: 
			accuracy_1 = 0
		else:
			accuracy_1 = correct_predictions_1 / predictions_1
		if total_instances_0 == 0:
			recall_0 = 0
		else:
			recall_0 = correct_predictions_0 / total_instances_0
		if total_instances_1 == 0:
			recall_1 = 0
		else:
			recall_1 = correct_predictions_1 / total_instances_1
		self.timeTotalPredict = time.time() - self.timeStart
		print("Predicción realizada en ",self.timeTotalPredict," segundos.")
		return total_instances_0, total_instances_1, accuracy, accuracy_0, accuracy_1, recall_0, recall_1

	def confusion_matrix_c45(self):
		# Calcula la matriz de confusión
		conf_matrix = confusion_matrix(self.y_test, self.y_pred)
		plt.figure(figsize=(8, 6))
		sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
		plt.title(f"Matriz de Confusión -  DT con ganancia mínima = {self.infoGainThreshold}")
		plt.xlabel('Predicción')
		plt.ylabel('Etiqueta Real')
		plt.show()

class Node:
	def __init__(self,isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []