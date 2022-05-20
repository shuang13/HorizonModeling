class Catalog():
	def __init__(self):
		self.base = "./data/"
		
		self.projectName = "迪北"
		self.layers = [
  		"J1a",
			"J1y",
			"J2kz1",
			"K1y",
			"TE_D",
			"TT_D"
		]
		self.faults = [
			"f_f1",# 0
			"f_f4",# 1
			"f_f5",# 2
			"f_f6",# 3
			"f1",# 4
			"f2",# 5
			"f2-1",# 6
			"f3",# 7
			"f4-1",# 8
			"f4",# 9
			"f6",# 10
			"f8",# 12
			"f7",# 11
			"f10_2017",# 13
			"f10_2020",# 14
			"f11",# 15
			"f12",# 16
			"f13",# 17
			"f15",# 18
			"f16",# 19
			"f17",# 20
			"f34",# 21
			"f70",# 22
		]
		# 层位相应断层的类型，0--无，-1--左边界，-2--右边界，1--左型（z型）逆断层,2--右型（反z）逆断层
		self.types = [
			#0  1  2  3  4  5  6  7  8  9  10 11 12 13  14 15 16 17 18 19 20 21 22
			[1, 1, 2, 0, 2, 3, 0, 1, 3, 4, 4, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1],# 'J1a'
			[0, 1, 2, 1, 2, 1, 0, 1, 3, 4, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1],# 'J1y'
			[0, 1, 2, 0, 1, 1, 1, 1, 3, 4, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, 1, 0, 1],# 'J2kz1'
			[0, 0, 0, 0, 2, 1, 1, 0, 1, 1, 0, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0],# 'K1y'
			[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],# 'TE_D'
			[1, 1, 2, 0, 2, 3, 0, 1, 3, 4, 4, 3, 4, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1],# 'TT_D'
		]
		