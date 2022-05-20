
class DrawDataManage():
	def __init__(self):
		self.minX = 99999
		self.maxX = -99999
		self.minY = 99999
		self.maxY = -99999
		self.minZ = 99999
		self.maxZ = -99999 
		self.xStart = 0
		self.xEnd = 0
		self.yStart = 0
		self.yEnd = 0
		self.zStart = 0
		self.zEnd = 0
		self.width = 800
		self.height = 800
		self.zDepth = 800
		self.colorList = [
  			"red",
  			"blue",
  			"green",
  			"yellow",
  			"orange",
  			"black",
		]
		self.layersData = []
		self.faultsData = []
		self.crossPointsData = []
		self.crossLinesData = []
		self.blockData = []

		
	def getLayersData(self):
		return self.layersData
	def getFaultsData(self):
		return self.faultsData
	def getCrossPointsData(self):
		return self.crossPointsData
	def getCrossLinesData(self):
		return self.crossLinesData
	def getBlockData(self):
		return self.crossBlockData
	def getMinX(self):
		return self.minX	
	def getMinY(self):
		return self.minY
	def getMinZ(self):
		return self.minZ
	def getMaxX(self):
		return self.maxX
	def getMaxY(self):
		return self.maxY
	def getMaxZ(self):
		return self.maxZ

	
	def setLayersData(self, data):
		self.layersData = data
	def setFaultsData(self, data):
		self.faultsData = data

	def setCrossPointsData(self, data):
		self.crossPointsData = data
	def setCrossLinesData(self, data):
		self.crossLinesData = data
	def setBlockData(self, data):
		self.blockData = data
	def setMinX(self, value):
		self.minX = value
	def setMinY(self, value):
		self.minY = value
	def setMinZ(self, value):
		self.minZ = value
	def setMaxX(self, value):
		self.maxX = value
	def setMaxY(self, value):
		self.maxY = value
	def setMaxZ(self, value):
		self.minZ = value