import DataDriver as dataDriver
from Catalog import Catalog
# 目录实例
catalog = Catalog()

class DataManage():
	def __init__(self):
		self.layersData = []
		self.faultsData = []
		self.crossPointsData = []
		self.crossLinesData = []
		self.bordersData = []
		self.vertices = []
	def setVertices(self, data):
		self.vertices = data
	def getVertices(self):
		return self.vertices
	def getLayersData(self):
		return self.layersData
	def getFaultsData(self):
		return self.faultsData
	def getCrossPointsData(self):
		return self.crossPointsData
	def getCrossLinesData(self):
		return self.crossLinesData
	def getBordersData(self):
		return self.bordersData
		
	def setLayersData(self, data):
		self.layersData = data
	def setFaultsData(self, data):
		self.faultsData = data

	def setCrossPointsData(self, data):
		self.crossPointsData = data
	def setCrossLinesData(self, data):
		self.crossLinesData = data

	def loadLayersData(self, filesName):
		flen = len(filesName)
		for i in range(flen):
			data = dataDriver.load_np(catalog.base + "预处理/层位/" + filesName[i] + ".txt")
			self.layersData.append(data)
	def loadFaultsData(self, filesName):
		flen = len(filesName)
		for i in range(flen):
			data = dataDriver.load_np(catalog.base + "预处理/断层/" + filesName[i] + ".txt")
			self.faultsData.append(data)
	def loadCrossLinesData(self, filesName):
		flen = len(filesName)
		for i in range(flen):
			data = dataDriver.load_np(filesName[i])
			self.crossLinesData.append(data)
	def loadCrossPointsData(self, types, layer, faults):
		flen = len(types)
		for i in range(flen):
			if(types[i] != 0):
				data1 = dataDriver.load_np_corss(catalog.base + "交线/" + layer + "-" + faults[i] + "-" + "L1" + ".txt")
				data2 = dataDriver.load_np_corss(catalog.base + "交线/" + layer + "-" + faults[i] + "-" + "L2" + ".txt")
			else:
				data1 = []
				data2 = []
			self.crossPointsData.append(data1)
			self.crossPointsData.append(data2)
