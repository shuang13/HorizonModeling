class Vertex:
	def __init__(self):
		self.index = -1
		self.branch = [-1, -1, -1, -1] # 上 下 左 右
		self.position = None
		# 是否为可用点
		self.live = 1
		# 是否为控制点
		self.control = 0


		# 0:原始层位点
		# 1:网格点，z = 0
		# 2:原始断层交点
		# 3:网格与断层交点，z = 0


		# 0:网格点，z = 0
		# 1:多层点
		# 2:网格与断层交点，z = 0
		# 3:交线点
		# 4:交线投影点
		self.type = 0 

		# 是否为边界点
		self.border = 0 
		# 该点存在于哪一个断层多边形，也就是序号，也有可能存在于多个断层多边形中
		self.last = []
		self.nextLineLastPoint = None
	def getNextLineLastPoint(self):
		return self.nextLineLastPoint
	def setNextLineLastPoint(self, c):
		self.nextLineLastPoint = c
	def getLast(self):
		return self.last
	def setLast(self, c):
		self.last = c

	def getType(self):
		return self.type
	def setType(self, c):
		self.type = c
	def getControl(self):
		return self.control
	def setControl(self, c):
		self.control = c	
	def getLive(self):
		return self.live
	def setLive(self, live):
		self.live = live
	def getBorder(self):
		return self.border
	def setBorder(self, border):
		self.border = border
	def setIndex(self, index):
		self.index = index

	def getIndex(self):
		return self.index
	def setPosition(self, position):
		self.position = position

	def getPosition(self):
		return self.position
	def getX(self):
		return self.position[0]
	def setX(self, v):
		self.position[0] = v
	def getY(self):
		return self.position[1]
	def setY(self, v):
		self.position[1] = v
	def getZ(self):
		return self.position[2]
	def setZ(self, v):
		self.position[2] = v
	def setHalfEdge(self, halfEdge):
		self.halfEdge = halfEdge

	def getHalfEdge(self):
		return self.halfEdge
	def setBranchTop(self, index):
		self.branch[0] = index
	def getBranchTop(self):
		return self.branch[0]
	def setBranchBottom(self, index):
		self.branch[1] = index
	def getBranchBottom(self):
		return self.branch[1]
	def setBranchLeft(self, index):
		self.branch[2] = index
	def getBranchLeft(self):
		return self.branch[2]
	def setBranchRight(self, index):
		self.branch[3] = index
	def getBranchRight(self):
		return self.branch[3]
	def setBranch(self, indexs):
		self.branch = indexs
	def getBranch(self):
		return self.branch

