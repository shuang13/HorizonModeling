'''
功能：地质层位建模
输入：层位、断层解释数据文件
输出：层位模型obj文件
'''
from DataManage import DataManage
from DrawData import DrawDataManage
from Catalog import Catalog
from Vertex import Vertex
import Utils as utils
import numpy as np
import matplotlib.pyplot as plt
import copy
# import matplotlib
# matplotlib.use("Agg")

# 水平向右相连
def linkR(p1, p2):
	Vertexs[p1].setBranchRight(p2)
	Vertexs[p2].setBranchLeft(p1)
# 垂直向上相连
def linkT(p1, p2):
	Vertexs[p1].setBranchTop(p2)
	Vertexs[p2].setBranchBottom(p1)
# 顶点向量转字符串
def p2str(arr):
	p = [str((arr[0] )), str((arr[1])), str((arr[2]))]
	p_str = ''.join(p)
	return p_str
# 比较两个顶点是否相等
def comparePoint(p1, p2):
	if(p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]):
		return True
	return False
# 求平均
def getMean(arr):
	sum = 0
	for i in range(len(arr)):
		sum = sum + arr[i]
	return sum / len(arr)
# 求中位数
def getMid(arr):
	l = len(arr)
	return arr[int(l / 2)]

def setGridZ(h, cross_pairs):
	for i in range(len(cross_pairs)):
		if(cross_pairs[i] == []):
			continue
		cross = cross_pairs[i]
		divideGridZ(h, cross)
	for i in range(len(h)):
		for j in range(len(h[i])):
			h_v = h[i][j]
			h_p = h_v.getPosition()
			for m in range(len(h_p[2])):
				if(len(h_p[2][m]) == 0):
					h_p[2][m] = m * 0.000000001
				else:
					mean = getMid(h_p[2][m])
					h_p[2][m] = mean
def divideGridZ(h, cross):
	k = 0
	for i in range(len(h)):
		if(k >= len(cross[0])):
			continue
		r1 = int(h[i][0].getPosition()[1] * 10000)
		r2 = int(cross[0][k][1] * 10000)
		t_p = cross[0][k]
		b_p = cross[1][k]
		l_p = []
		r_p = []
		if(t_p[0] < b_p[0]):
			l_p = t_p
			r_p = b_p
		else:
			l_p = b_p
			r_p = t_p
		if(r1 == r2):
			k = k + 1
		else:
			continue
		for j in range(len(h[i])):
			h_v = h[i][j]
			h_p = h_v.getPosition()
			
			
			r1 = h_p[0]
			r2 = l_p[0]
			r3 = r_p[0]
			if(r1 > r2 and r1 < r3):
				mid_z = (l_p[2] + r_p[2]) / 2
				l = len(h_p[2])
				if(len(h_p[2][0]) == 0):
						h_p[2].append([])
				for m in range(l):
					temp = []
					temp2 = []
					for n in range(len(h_p[2][m])):
						if(h_p[2][m][n] < mid_z):
							temp.append(h_p[2][m][n])
						else:
							temp2.append(h_p[2][m][n])
					if(len(temp) != 0 and len(temp2) != 0):
						h_p[2][m] = temp
						h_p[2].append(temp2)
					elif(len(temp) == 0 and len(temp2) != 0):
						h_p[2][m] = temp2
					elif(len(temp) != 0 and len(temp2) == 0):
						h_p[2][m] = temp
				

if __name__ == '__main__':
	# 平滑次数
	smooth_times = 30
	catalog = Catalog()
	'''
	数据加载	
	'''
	for q in range(0, 5):
		############################# 数据预处理 ###########################
		dataManage = DataManage()
		drawDataManage = DrawDataManage()
		dataManage.loadLayersData([catalog.layers[q]])
		dataManage.loadFaultsData(catalog.faults)
		dataManage.loadCrossPointsData(catalog.types[q], catalog.layers[q], catalog.faults)
		newLayer = []
		newf = []
		TYPES = catalog.types[q]
		layersData = dataManage.getLayersData()
		faultsData = dataManage.getFaultsData()


		for i in range(len(layersData[0])):
			if(layersData[0][i][0] >= 275):
				newLayer.append(layersData[0][i])
		layersData = [newLayer]
		newf = []
		for i in range(len(faultsData)):
			temp = []
			for j in range(len(faultsData[i])):
				if(faultsData[i][j][0] >= 275):
					temp.append(faultsData[i][j])
			newf.append(temp)
		faultsData = newf
		crossPointsData = dataManage.getCrossPointsData()

		# 交线首尾封闭
		for i in range(int(len(crossPointsData) / 2)):
			if(TYPES[i] <= 0):
				continue
			if(crossPointsData[i * 2][0][1] > 176 and not((q == 5 and i == 10) or (q == 0 and i == 10) or (q == 1 and i == 10))):
				x = (crossPointsData[i * 2][0][0] + crossPointsData[i * 2 + 1][0][0]) / 2
				z = (crossPointsData[i * 2][0][2] + crossPointsData[i * 2 + 1][0][2]) / 2
				add_p = [x, crossPointsData[i * 2][0][1] - 16, z]
				crossPointsData[i * 2].insert(0, add_p)
				crossPointsData[i * 2 + 1].insert(0, add_p)
			if(crossPointsData[i * 2][-1][1] < 1232):
				x = (crossPointsData[i * 2][-1][0] + crossPointsData[i * 2 + 1][-1][0]) / 2
				z = (crossPointsData[i * 2][-1][2] + crossPointsData[i * 2 + 1][-1][2]) / 2
				add_p = [x, crossPointsData[i * 2][-1][1] + 16, z]
				crossPointsData[i * 2].append(add_p)
				crossPointsData[i * 2 + 1].append(add_p)

		layersLen = len(layersData)
		faultsLen = len(faultsData)
		crossPointsLen = len(crossPointsData)
	
		blocksData = layersData + faultsData + crossPointsData
	
		# 工区边界
		minX = drawDataManage.getMinX()
		maxX = drawDataManage.getMaxX()
		minY = drawDataManage.getMinY()
		maxY = drawDataManage.getMaxY()
		minZ = drawDataManage.getMinZ()
		maxZ = drawDataManage.getMaxZ() 
	
		for i in range(len(blocksData)):
			if(blocksData[i] == []):
				continue
			npBlocks = np.array(blocksData[i])
			npBlocks_T = npBlocks.T
			x11 = npBlocks_T[0]
			y11 = npBlocks_T[1]
			z11 = npBlocks_T[2]
			if(minX > np.min(x11)):
				minX = np.min(x11)
			if(maxX < np.max(x11)):
				maxX = np.max(x11)
			if(minY > np.min(y11)):
				minY = np.min(y11)
			if(maxY < np.max(y11)):
				maxY = np.max(y11)	
			if(minZ > np.min(z11)):
				minZ = np.min(z11)
			if(maxZ < np.max(z11)):
				maxZ = np.max(z11)
		drawDataManage.setMinX(minX)
		drawDataManage.setMaxX(maxX)
		drawDataManage.setMinY(minY)
		drawDataManage.setMaxY(maxY)
		drawDataManage.setMinZ(minZ)
		drawDataManage.setMaxZ(maxZ) 

		# 归一化
		norm_blockData = []
		for i in range(len(blocksData)):
			if(blocksData[i] == []):
				norm_blockData.append([])
				continue
			npBlocks = np.array(blocksData[i])
			npBlocks_T = npBlocks.T
			x11 = npBlocks_T[0]
			y11 = npBlocks_T[1]
			z11 = npBlocks_T[2]

			x11 = utils.normalize(x11, minX, maxX)
			y11 = utils.normalize(y11, minY, maxY)
			z11 = utils.normalize(z11, minZ, maxZ)
			norm_blockData.append([x11, y11, z11])
		drawDataManage.setBlockData(norm_blockData)
		norm_layers = []
		norm_faults = []
		norm_crossPoints = []
		for i in range(len(norm_blockData)):
			if(i < layersLen):
				norm_layers.append(norm_blockData[i])
			if(i >= layersLen and i < layersLen + faultsLen):
				norm_faults.append(norm_blockData[i])
			if(i >= layersLen + faultsLen):
				norm_crossPoints.append(norm_blockData[i])
		drawDataManage.setLayersData(norm_layers)
		drawDataManage.setFaultsData(norm_faults)
		drawDataManage.setCrossPointsData(norm_crossPoints)

	############################# 数据准备完成 ###########################

		# 三分量
		sample_faults = drawDataManage.getFaultsData()
		sample_layers = drawDataManage.getLayersData()
		sample_crossPoints = drawDataManage.getCrossPointsData()
	
		sample_crossPoints_len = len(sample_crossPoints)
		sample_crossPoints_pairs = []
		# 一个断层对一个层位有两条交线
		pair = []
		for i in range(sample_crossPoints_len):
			pair.append(sample_crossPoints[i])
			if((i % 2) == 1):
				sample_crossPoints_pairs.append(pair)
				pair = []

	############################# 层位处理 ###############################
	
		sample_layer = sample_layers[0] # 三分量
		sample_layer_position = np.array(sample_layer).T
		sample_layer_trace = utils.dataToTraceData(sample_layer_position)

		sample_layer_len = len(sample_layer_position)

		# 网格
		beishu = 2
		x_grid_size = 50 * beishu
		y_grid_size = 66 * beishu
		grid_x = np.linspace(0, 1, x_grid_size + 1)
		grid_y = np.linspace(0, 1, y_grid_size + 1)

		crossPoints_grid_pairs = []

		# 交线按照网格采样
		for i in range(len(sample_crossPoints_pairs)):
			if(TYPES[i] == 0):
				crossPoints_grid_pairs.append([])
				continue
			sample_crossPoints_pair = sample_crossPoints_pairs[i]
			top = sample_crossPoints_pair[0]
			bottom = sample_crossPoints_pair[1]
			crossPoints_top_grid = []
			crossPoints_bottom_grid = []
			for j in range(len(top[0]) - 1):
				for k in range(beishu):
					crossPoints_top_grid.append([top[0][j] + (top[0][j + 1] - top[0][j]) / beishu * k, top[1][j] + (top[1][j + 1] - top[1][j]) / beishu * k,  top[2][j] + (top[2][j + 1] - top[2][j]) / beishu * k])
					crossPoints_bottom_grid.append([bottom[0][j] + (bottom[0][j + 1] - bottom[0][j]) / beishu * k, bottom[1][j] + (bottom[1][j + 1] - bottom[1][j]) / beishu * k,  bottom[2][j] + (bottom[2][j + 1] - bottom[2][j]) / beishu * k])
			crossPoints_top_grid.append([top[0][j + 1], top[1][j + 1], top[2][j + 1]])
			crossPoints_bottom_grid.append([bottom[0][j + 1], bottom[1][j + 1], bottom[2][j + 1]])
			crossPoints_grid_pairs.append([crossPoints_top_grid, crossPoints_bottom_grid])

		# 网格拓扑构建
		grid_vertices = []
		grid_vertices_grid = []
		v_index = -1
		# 初始化网格
		for i in range(len(grid_y)):
			grid_vertices_temp = []
			for j in range(len(grid_x)):
				v_index = v_index + 1
				v = Vertex()
				v.setIndex(v_index)
				v.setPosition([grid_x[j], grid_y[i] , [[]]])
				v.setType(0)
				if(j - 1 < 0):
					v.setBranchLeft(-1)
				else:
					v.setBranchLeft(v_index - 1)

				if(j + 1 > len(grid_x) - 1):
					v.setBranchRight(-1)

				else:
					v.setBranchRight(v_index + 1)

				if(i - 1 < 0):
					v.setBranchBottom(-1)

				else:
					v.setBranchBottom(v_index - len(grid_x))

				if(i + 1 > len(grid_y) - 1):
					v.setBranchTop(-1)

				else:
					v.setBranchTop(v_index + len(grid_x))

				grid_vertices.append(v)
				grid_vertices_temp.append(v)
			grid_vertices_grid.append(grid_vertices_temp)

		# 根据种子数据设置网格z值
		for j in range(len(sample_layer_position)):
			p = sample_layer_position[j]
			p_x = p[0]
			p_y = p[1]
			p_z = p[2]
			grid_i = int(p_x * x_grid_size + 0.5)
			grid_j = int(p_y * y_grid_size + 0.5)
			grid_vertices[grid_j * (x_grid_size + 1) + grid_i].position[2][0].append(p_z)
			grid_vertices[grid_j * (x_grid_size + 1) + grid_i].setControl(0)

		###### 将网格点的z值按照原始数据进行分层，将区块内的点z值求和平均采样
		setGridZ(grid_vertices_grid, crossPoints_grid_pairs)
		
		faces = []
		vertices_hash = {}
		vertices = []
		vertices_index = -1
		Vertexs = []
		# 网格vertex点创建
		for i in range(len(grid_vertices_grid)):
			for j in range(len(grid_vertices_grid[i])):
				h_v = grid_vertices_grid[i][j]
				h_p = h_v.getPosition()
				if(len(h_p[2]) == 1):
					h_v.setType(0)
				if(len(h_p[2]) == 2):
					h_v.setType(1)
				if(len(h_p[2]) == 3):
					h_v.setType(10)
				for m in range(len(h_p[2])):
					
					vertices_index = vertices_index + 1
					p = [str((h_p[0] )), str((h_p[1])), str((h_p[2][m]))]
					p_str = ''.join(p)
					vertices_hash[p_str] = vertices_index

					v = Vertex()
					v.setIndex(v_index)
					if(h_p[2][m] > 0.00000001):
						v.setControl(1)
					else:
						v.setControl(0)
					if(len(h_p[2]) == 1):
						v.setType(0)
					else:
						v.setType(1)

					v.setPosition([h_p[0], h_p[1], h_p[2][m]])
					Vertexs.append(v)
					
		# 交线vertex点创建
		cross_border_index = []
		
		for i in range(len(crossPoints_grid_pairs)):
			if(TYPES[i] == 0):
				continue
			cross_border_index_temp = []
			for j in range(len(crossPoints_grid_pairs[i][0])):
				t_p = crossPoints_grid_pairs[i][0][j]
				b_p = crossPoints_grid_pairs[i][1][j]
				l_p = b_p
				r_p = t_p
				if(t_p[0] < b_p[0]):
					l_p = t_p
					r_p = b_p
				if(comparePoint(t_p, b_p) and TYPES[i] > 0):
					p = [str((t_p[0] )), str((t_p[1])), str(t_p[2])]
					p_str = ''.join(p)
					vertices_index = vertices_index + 1
					vertices_hash[p_str] = vertices_index
					v = Vertex()
					v.setIndex(v_index)
					v.setPosition([t_p[0], t_p[1], t_p[2]])
					v.setControl(1)
					v.setType(3)
					Vertexs.append(v)
					cross_border_index_temp.append(j)
					cross_border_index.append(cross_border_index_temp)
					continue

				p1 = [str((t_p[0] )), str((t_p[1])), str(0.01)]
				p2 = [str((t_p[0] )), str((t_p[1])), str((t_p[2]))]

				p3 = [str((b_p[0] )), str((b_p[1])), str(0.01)]
				p4 = [str((b_p[0] )), str((b_p[1])), str((b_p[2]))]
				p_str1 = ''.join(p1)
				p_str2 = ''.join(p2)
				p_str3 = ''.join(p3)
				p_str4 = ''.join(p4)
				vertices_index = vertices_index + 1
				vertices_hash[p_str1] = vertices_index
				v = Vertex()
				v.setIndex(v_index)
				v.setPosition([t_p[0], t_p[1], 0.01])
				v.setControl(0)
				v.setType(4)
				Vertexs.append(v)

				vertices_index = vertices_index + 1
				vertices_hash[p_str2] = vertices_index
				v = Vertex()
				v.setIndex(v_index)
				v.setPosition([t_p[0], t_p[1], t_p[2]])
				v.setControl(1)
				v.setType(3)
				Vertexs.append(v)

				vertices_index = vertices_index + 1
				vertices_hash[p_str3] = vertices_index
				v = Vertex()
				v.setIndex(v_index)
				v.setPosition([b_p[0], b_p[1], 0.01])
				v.setControl(0)
				v.setType(4)
				Vertexs.append(v)

				vertices_index = vertices_index + 1
				vertices_hash[p_str4] = vertices_index

				v = Vertex()
				v.setIndex(v_index)
				v.setPosition([b_p[0], b_p[1], b_p[2]])
				v.setType(3)
				v.setControl(1)
				Vertexs.append(v)
		x = []
		y = []
		z = []
		x2 = []
		y2 = []
		z2 = []
		x3 = []
		y3 = []
		z3 = []
		borders_faults = []
		p_live = 2
		f8_cross = []
		f6_cross = []
		f4_cross = []
		f2_cross = []
		for u in range(len(crossPoints_grid_pairs)):
			if(TYPES[u] == 0):
				continue

			# 左边界断层
			if(TYPES[u] == -1):
				borders_faults.append(crossPoints_grid_pairs[u][0])
			# 右边界断层
			if(TYPES[u] == -2):
				borders_faults.append(crossPoints_grid_pairs[u][0])
			top = crossPoints_grid_pairs[u][1]
			bottom = crossPoints_grid_pairs[u][0]

			cross_index = -1
			border_flag = 0
			p_live = p_live + 1
			if(u == 13):
				p_live = 100
			f8_i = 0
			f4_i = 0
			f6_i = 0
			f2_i = 0
			f8_flag = 0
			f6_flag = 0
			f4_flag = 0
			f2_flag = 0
			for i in range(len(grid_vertices_grid) - 1):
				a = int(grid_vertices_grid[i][0].getPosition()[1] * 10000)
				b = int(top[-2][1] * 10000)
				c = int(top[0][1] * 10000)
				

				if((comparePoint(top[0], bottom[0]) and a == c)):
					border_flag = 1
				elif((comparePoint(top[-1], bottom[-1]) and a == b)):
					border_flag = -1
				else:
					border_flag = 0
				if(TYPES[u] == -1):
					border_flag = 0
				if(a >= c and a <= b):
					cross_index = cross_index + 1
					t_p = top[cross_index]
					b_p = bottom[cross_index]

					t_p2 = top[cross_index + 1]
					b_p2 = bottom[cross_index + 1]
					l_p = []
					r_p = []
					l_p2 = []
					r_p2 = []
					if(t_p[0] < b_p[0]):
						l_p = t_p
						r_p = b_p
					else:
						l_p = b_p
						r_p = t_p
					if(t_p2[0] < b_p2[0]):
						l_p2 = t_p2
						r_p2 = b_p2
					else:
						l_p2 = b_p2
						r_p2 = t_p2
					if(l_p[0] > l_p2[0]):
						l_p_min = l_p2
						l_p_max = l_p
					else:
						l_p_min = l_p
						l_p_max = l_p2

					if(l_p[1] > l_p2[1]):
						temp = l_p
						l_p = l_p2
						l_p2 = temp
					if(r_p[0] < r_p2[0]):
						r_p_max = r_p2
						r_p_min = r_p
					else:
						r_p_max = r_p
						r_p_min = r_p2
					if(r_p[1] > r_p2[1]):
						temp = r_p
						r_p = r_p2
						r_p2 = temp
					
					# TT_D的f8断层交线
					if(q == 5 and u == 11):
						f8_cross.append([l_p, r_p])
					if(q == 5 and u == 9):
						f4_cross.append([l_p, r_p])
					if(q == 5 and u == 10):
						f6_cross.append([l_p, r_p])
				
					if(q == 5 and u == 12):
						if(f8_i < len(f8_cross) - 1 and l_p[1] == f8_cross[f8_i][0][1]):
							f8_flag = 1
							f8_i = f8_i + 1
						if(f4_i < len(f4_cross) - 1 and l_p[1] == f4_cross[f4_i][0][1]):
							f4_flag = 1
							f4_i = f4_i + 1
						if(f6_i < len(f6_cross) - 1 and l_p[1] == f6_cross[f6_i][0][1]):
							f6_flag = 1
							f6_i = f6_i + 1
					top_point = True
					for j in range(len(grid_vertices_grid[i]) - 1):
						h_v = grid_vertices_grid[i][j]
						
						h_p = h_v.getPosition()
						
						if(len(h_p[2]) == 2):
							x2.append(h_p[0])
							y2.append(h_p[1])
						elif(len(h_p[2]) == 1):
							x.append(h_p[0])
							y.append(h_p[1])
						elif(len(h_p[2]) == 3):
							x3.append(h_p[0])
							y3.append(h_p[1])
						v1 = grid_vertices_grid[i][j]
						v2 = grid_vertices_grid[i][j + 1]
						v3 = grid_vertices_grid[i + 1][j]
						v4 = grid_vertices_grid[i + 1][j + 1]

						p1 = v1.getPosition()
						p2 = v2.getPosition()
						p3 = v3.getPosition()
						p4 = v4.getPosition()
						pre_p1 = grid_vertices_grid[i][j - 1].getPosition()
						pre_p2 = grid_vertices_grid[i + 1][j - 1].getPosition()
					
						ps = [p1, p2, p3, p4]
						pi = []
						# 交线头部闭合
						if(border_flag == 1):
							p1_i = vertices_hash[p2str([p1[0], p1[1], p1[2][0]])]
							p2_i = vertices_hash[p2str([p2[0], p2[1], p2[2][0]])]
							p3_i = vertices_hash[p2str([p3[0], p3[1], p3[2][0]])]
							p4_i = vertices_hash[p2str([p4[0], p4[1], p4[2][0]])]
							pre_p1_i = vertices_hash[p2str([pre_p1[0], pre_p1[1], pre_p1[2][0]])]
							pre_p2_i = vertices_hash[p2str([pre_p2[0], pre_p2[1], pre_p2[2][0]])]

							# 交线在一个格子
							if(p1[0] < l_p[0] and p3[0] < l_p2[0] and p2[0] > r_p[0] and p4[0] > r_p2[0]):
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								l_c_pp1 = [l_p2[0], l_p2[1], l_p2[2]]
								l_c_pp2 = [l_p2[0], l_p2[1], 0.01]

								o_c_pp = [l_p[0], l_p[1], l_p[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p2[0], r_p2[1], r_p2[2]]
								r_c_pp2 = [r_p2[0], r_p2[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

								
								if(len(p2[2]) == 1 and len(p4[2]) == 1): 
									linkR(o_c_pp_i, p2_i)
									linkR(r_c_pp2_i, p4_i)

									faces.append([p2_i, p4_i, r_c_pp2_i])
									faces.append([r_c_pp2_i, o_c_pp_i, p2_i])
								if(len(p1[2]) == 1 and len(p3[2]) == 1): 
									linkR(p3_i, l_c_pp2_i)
									linkR(p1_i, o_c_pp_i)
									faces.append([p1_i, p3_i, o_c_pp_i])
									faces.append([p3_i, o_c_pp_i, l_c_pp2_i])

								# 多层部分
								faces.append([l_c_pp1_i, r_c_pp2_i, o_c_pp_i])
								faces.append([l_c_pp2_i, r_c_pp1_i, o_c_pp_i])
								if(len(p1[2]) == 1 and len(p2[2]) == 1): 
									faces.append([p1_i, o_c_pp_i, p2_i])
						
						
							# 交线在两个格子,重复点位p3
							elif(p1[0] < l_p[0] and p2[0] > r_p[0] and p3[0] > l_p2[0] and p4[0] > r_p2[0] and p3[0] < r_p2[0]):
								p3_2_i = vertices_hash[p2str([p3[0], p3[1], p3[2][1]])]

								l_c_pp1 = [l_p2[0], l_p2[1], l_p2[2]]
								l_c_pp2 = [l_p2[0], l_p2[1], 0.01]

								o_c_pp = [l_p[0], l_p[1], l_p[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p2[0], r_p2[1], r_p2[2]]
								r_c_pp2 = [r_p2[0], r_p2[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

								linkR(p1_i, o_c_pp_i)
								linkR(o_c_pp_i, p2_i)
								linkR(l_c_pp2_i, p3_2_i)
								linkR(l_c_pp1_i, p3_i)
								linkR(p3_2_i, r_c_pp1_i)
								linkR(p3_i, r_c_pp2_i)
								linkR(r_c_pp2_i, p4_i)
								linkR(pre_p2_i, l_c_pp2_i)
								linkR(pre_p1_i, p1_i)

								faces.append([l_c_pp2_i, o_c_pp_i, p3_2_i])
								faces.append([l_c_pp1_i, o_c_pp_i, p3_i])
								faces.append([p3_2_i, o_c_pp_i, r_c_pp1_i])
								faces.append([p3_i, o_c_pp_i, r_c_pp2_i])
								faces.append([r_c_pp2_i, o_c_pp_i, p2_i])
								faces.append([r_c_pp2_i, p4_i, p2_i])
								faces.append([p1_i, o_c_pp_i, l_c_pp2_i])
								faces.append([pre_p1_i, pre_p2_i, l_c_pp2_i])
								faces.append([pre_p1_i, p1_i, l_c_pp2_i])
								faces.append([p1_i, o_c_pp_i, p2_i])
							# 交线在两个格子,重复点位p3
							elif(pre_p1[0] < l_p[0] and p1[0] > l_p[0] and pre_p2[0] < l_p2[0] and p3[0] > l_p2[0] and p3[0] < r_p2[0] and p1[0] > r_p[0] and p4[0] > r_p2[0]):
								p3_2_i = vertices_hash[p2str([p3[0], p3[1], p3[2][1]])]

								l_c_pp1 = [l_p2[0], l_p2[1], l_p2[2]]
								l_c_pp2 = [l_p2[0], l_p2[1], 0.01]

								o_c_pp = [l_p[0], l_p[1], l_p[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p2[0], r_p2[1], r_p2[2]]
								r_c_pp2 = [r_p2[0], r_p2[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

								linkR(pre_p2_i, l_c_pp2_i)
								linkR(l_c_pp2_i, p3_2_i)
								linkR(l_c_pp1_i, p3_i)
								linkR(p3_2_i, r_c_pp1_i)
								linkR(p3_i, r_c_pp2_i)
								linkR(r_c_pp2_i, p4_i)
								linkR(pre_p1_i, o_c_pp_i)
								linkR(o_c_pp_i, p1_i)
								linkR(p1_i, p2_i)
								faces.append([l_c_pp2_i, o_c_pp_i, p3_2_i])
								faces.append([l_c_pp1_i, o_c_pp_i, p3_i])
								faces.append([p3_2_i, o_c_pp_i, r_c_pp1_i])
								faces.append([p3_i, o_c_pp_i, r_c_pp2_i])
								faces.append([r_c_pp2_i, o_c_pp_i, p1_i])
								faces.append([r_c_pp2_i, p1_i, p2_i])
								faces.append([r_c_pp2_i, p2_i, p4_i])
								faces.append([pre_p2_i, pre_p1_i, l_c_pp2_i])
								faces.append([pre_p1_i, o_c_pp_i, l_c_pp2_i])
								faces.append([pre_p1_i, o_c_pp_i, p1_i])
						
							elif(p1[0] < l_p[0] and p2[0] > r_p[0] and p4[0] < r_p2[0] and pre_p2[0] > l_p2[0]):
								pre_p3 = grid_vertices_grid[i][j - 2].getPosition()
								pre_p4 = grid_vertices_grid[i + 1][j - 2].getPosition()
								pre_p3_i = vertices_hash[p2str([pre_p3[0], pre_p3[1], pre_p3[2][0]])]
								pre_p4_i = vertices_hash[p2str([pre_p4[0], pre_p4[1], pre_p4[2][0]])]

								next_p3 = grid_vertices_grid[i][j + 2].getPosition()
								next_p4 = grid_vertices_grid[i + 1][j + 2].getPosition()
								next_p3_i = vertices_hash[p2str([next_p3[0], next_p3[1], next_p3[2][0]])]
								next_p4_i = vertices_hash[p2str([next_p4[0], next_p4[1], next_p4[2][0]])]
								p3_2_i = vertices_hash[p2str([p3[0], p3[1], p3[2][1]])]
								p4_2_i = vertices_hash[p2str([p4[0], p4[1], p4[2][1]])]
								pre_p2_2_i = vertices_hash[p2str([pre_p2[0], pre_p2[1], pre_p2[2][1]])]

								l_c_pp1 = [l_p2[0], l_p2[1], l_p2[2]]
								l_c_pp2 = [l_p2[0], l_p2[1], 0.01]

								o_c_pp = [l_p[0], l_p[1], l_p[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p2[0], r_p2[1], r_p2[2]]
								r_c_pp2 = [r_p2[0], r_p2[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
								linkR(pre_p4_i, l_c_pp2_i)

								linkR(l_c_pp2_i, pre_p2_2_i)
								linkR(l_c_pp1_i, pre_p2_i)
								linkR(pre_p2_2_i, p3_2_i)
								linkR(pre_p2_i, p3_i)
								linkR(p3_2_i, p4_2_i)
								linkR(p3_i, p4_i)
								linkR(p4_2_i, r_c_pp1_i)
								linkR(p4_i, r_c_pp2_i)
								linkR(r_c_pp2_i, next_p4_i)
								linkR(pre_p3_i, pre_p1_i)
								linkR(pre_p1_i, p1_i)
								linkR(p1_i, o_c_pp_i)
								# linkR(o_c_pp_i, p2_i)
								# linkR(p2_i, p3_i)

								faces.append([pre_p3_i, pre_p4_i, pre_p1_i])
								faces.append([pre_p1_i, pre_p4_i, l_c_pp2_i])
								faces.append([pre_p1_i, p1_i, l_c_pp2_i])
								faces.append([o_c_pp_i, p1_i, l_c_pp2_i])
								# faces.append([next_p3_i, next_p4_i, p2_i])
								# faces.append([r_c_pp2_i, next_p4_i, p2_i])
								# faces.append([r_c_pp2_i, o_c_pp_i, p2_i])

								faces.append([pre_p2_2_i, o_c_pp_i, l_c_pp2_i])
								faces.append([pre_p2_i, o_c_pp_i, l_c_pp1_i])
								faces.append([pre_p2_2_i, o_c_pp_i, p3_2_i])
								faces.append([pre_p2_i, o_c_pp_i, p3_i])
								faces.append([p4_2_i, o_c_pp_i, p3_2_i])
								faces.append([p4_i, o_c_pp_i, p3_i])
								# faces.append([p4_2_i, o_c_pp_i, r_c_pp1_i])
								# faces.append([p4_i, o_c_pp_i, r_c_pp2_i])
							else:
								if(q == 5 and u != 12):
									continue

								l_c_pp1 = [l_p2[0], l_p2[1], l_p2[2]]
								l_c_pp2 = [l_p2[0], l_p2[1], 0.01]

								o_c_pp = [l_p[0], l_p[1], l_p[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p2[0], r_p2[1], r_p2[2]]
								r_c_pp2 = [r_p2[0], r_p2[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
								if(top_point and p3[0] > l_p2[0] and p4[0] < r_p2[0]):
									k = 1
									t1 = grid_vertices_grid[i + 1][j + k].getPosition()
									t2 = grid_vertices_grid[i + 1][j].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
									t2_2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][1]])]
									linkR(l_c_pp2_i, t2_2_i)
									linkR(l_c_pp1_i, t2_i)
									faces.append([l_c_pp2_i, t2_2_i, o_c_pp_i])
									faces.append([l_c_pp1_i, t2_i, o_c_pp_i])

									while(t1[0] < r_p2[0]):
										t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
										t1_2_i = vertices_hash[p2str([t1[0], t1[1], t1[2][1]])]
										t2_2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][1]])]
										linkR(t1_i, t2_i)
										linkR(t1_2_i, t2_2_i)
										linkT(t1_i, o_c_pp_i)
										linkT(t2_i, o_c_pp_i)
										linkT(t1_2_i, o_c_pp_i)
										linkT(t2_2_i, o_c_pp_i)
										faces.append([t1_i, t2_i, o_c_pp_i])
										faces.append([t1_2_i, t2_2_i, o_c_pp_i])

										t2 = copy.deepcopy(t1)
										k = k + 1

										t1 = grid_vertices_grid[i + 1][j + k].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
									t2_2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][1]])]
									linkR(t2_i, r_c_pp2_i)
									linkR(t2_2_i, r_c_pp1_i)
	
									faces.append([t2_i, r_c_pp2_i, o_c_pp_i])
									faces.append([t2_2_i, r_c_pp1_i, o_c_pp_i])
									faces.append([p1_i, p2_i, o_c_pp_i])
									top_point = False
								# 网格单层与左交线拼接处
								if(p1[0] < l_p_min[0] and p2[0] > l_p_min[0]):
									if(TYPES[u] == 4 and  v1.getType() == 10):
											continue
										
									pp1 = [p1[0], p1[1], p1[2][0]]
									pp2 = [p2[0], p2[1], p2[2][0]]
									pp3 = [p3[0], p3[1], p3[2][0]]
									pp4 = [p4[0], p4[1], p4[2][0]]
									pp1_i = vertices_hash[p2str(pp1)]
									pp2_i = vertices_hash[p2str(pp2)]
									pp3_i = vertices_hash[p2str(pp3)]
									pp4_i = vertices_hash[p2str(pp4)]

									c_pp1 = [l_p[0], l_p[1],l_p[2]]
									c_pp2 = [l_p2[0], l_p2[1], 0.01]

									c_pp1_i = vertices_hash[p2str(c_pp1)]
									c_pp2_i = vertices_hash[p2str(c_pp2)]


									next_p1 = grid_vertices_grid[i][j + 2].getPosition()
									next_p2 = grid_vertices_grid[i + 1][j + 2].getPosition()
									pre_p1 = grid_vertices_grid[i][j - 1].getPosition()
									pre_p2 = grid_vertices_grid[i + 1][j - 1].getPosition()
									# 交线在一个格子内
									if(p2[0] > l_p_max[0] and p4[0] > l_p_max[0]):
										linkR(pp1_i, c_pp1_i)
										linkR(pp3_i, c_pp2_i)
										linkT(c_pp1_i, c_pp2_i)

										faces.append([pp1_i, pp3_i, c_pp1_i])
										faces.append([c_pp1_i, c_pp2_i, pp3_i])
									# 交线在两个格子以上
									elif(p1[0] < l_p[0] and p2[0] > l_p[0] and p4[0] < l_p2[0]):
										k = 1
										linkR(pp1_i, c_pp1_i)
										linkR(pp3_i, pp4_i)
										linkT(pp1_i, pp3_i)

										faces.append([pp1_i, pp3_i, pp4_i])
										faces.append([pp1_i, c_pp1_i, pp4_i])
										t1 = grid_vertices_grid[i + 1][j + k + 1].getPosition()
										t2 = grid_vertices_grid[i + 1][j + 1].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

										while(t1[0] < l_p2[0]):
											t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
											t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

											linkR(t2_i, t1_i)
											faces.append([t1_i, t2_i, c_pp1_i])

											t2 = copy.deepcopy(t1)
											k = k + 1

											t1 = grid_vertices_grid[i + 1][j + k].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
										linkR(t2_i, c_pp2_i)
										faces.append([c_pp1_i, t2_i, c_pp2_i])
									# 交线在两个格子以上反
									elif(p3[0] < l_p2[0] and p4[0] > l_p2[0] and p2[0] < l_p[0]):
										k = 1
										linkR(pp3_i, c_pp2_i)
										linkR(pp1_i, pp2_i)
										linkT(pp1_i, pp3_i)
										faces.append([pp1_i, pp2_i, pp3_i])
										faces.append([pp2_i, c_pp2_i, pp3_i])
										t1 = grid_vertices_grid[i][j + k + 1].getPosition()
										t2 = grid_vertices_grid[i][j + 1].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

										while(t1[0] < l_p[0]):
											t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
											t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

											linkR(t2_i, t1_i)
											linkT(t2_i, c_pp2_i)
											faces.append([t1_i, t2_i, c_pp2_i])

											t2 = copy.deepcopy(t1)
											k = k + 1

											t1 = grid_vertices_grid[i][j + k].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
										linkR(t2_i, c_pp1_i)
										faces.append([c_pp1_i, t2_i, c_pp2_i])
								# 网格单层与右交线拼接处
								if(p1[0] < r_p_max[0] and p2[0] > r_p_max[0]):
									if(TYPES[u] == 4 and  v1.getType() == 10):
											continue
									if(len(p2[2]) == 2):
										continue
									if(len(p4[2]) == 2): 
										continue
									if(TYPES[u] == -1):
										continue
									pp1 = [p1[0], p1[1], p1[2][0]]
									pp2 = [p2[0], p2[1], p2[2][0]]
									pp3 = [p3[0], p3[1], p3[2][0]]
									pp4 = [p4[0], p4[1], p4[2][0]]
									pp1_i = vertices_hash[p2str(pp1)]
									pp2_i = vertices_hash[p2str(pp2)]
									pp3_i = vertices_hash[p2str(pp3)]
									pp4_i = vertices_hash[p2str(pp4)]
									pre_p1 = grid_vertices_grid[i][j - 1].getPosition()
									pre_p2 = grid_vertices_grid[i + 1][j - 1].getPosition()
									pre_p3 = grid_vertices_grid[i][j - 2].getPosition()
									pre_p4 = grid_vertices_grid[i + 1][j - 2].getPosition()
									dis_min = 9999

									c_pp1 = [r_p[0], r_p[1], r_p[2]]
									c_pp2 = [r_p2[0], r_p2[1], 0.01]

									c_pp1_i = vertices_hash[p2str(c_pp1)]
									c_pp2_i = vertices_hash[p2str(c_pp2)]

									# 交线在一个格子内
									if(p2[0] > r_p[0] and p4[0] > r_p2[0] and p3[0] < r_p2[0] and p1[0] < r_p[0]):
										linkR(c_pp1_i, pp2_i)
										linkR(c_pp2_i, pp4_i)
										linkT(c_pp1_i, c_pp2_i)

										faces.append([pp2_i, pp4_i, c_pp2_i])
										faces.append([c_pp1_i, c_pp2_i, pp2_i])
									# 交线在两个格子以上
									elif(p3[0] < r_p2[0] and p4[0] > r_p2[0] and p1[0] > r_p[0]):
										k = 1
										linkR(c_pp2_i, pp4_i)
										linkR(pp1_i, pp2_i)
										linkT(pp2_i, pp4_i)

										faces.append([pp1_i, pp2_i, pp4_i])
										faces.append([pp1_i, c_pp2_i, pp4_i])
										t1 = grid_vertices_grid[i][j - k].getPosition()
										t2 = grid_vertices_grid[i][j].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

										while(t1[0] > r_p[0]):
											t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
											t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

											linkR(t1_i, t2_i)
											linkT(t2_i, c_pp2_i)

											faces.append([t1_i, t2_i, c_pp2_i])

											t2 = copy.deepcopy(t1)
											k = k + 1

											t1 = grid_vertices_grid[i][j - k].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
										linkR(c_pp1_i, t2_i)
										faces.append([c_pp1_i, t2_i, c_pp2_i])
									# 交线在两个格子以上反
									elif(p1[0] < r_p[0] and p2[0] > r_p[0] and p3[0] > r_p2[0]):
										k = 1
										linkR(c_pp1_i, pp2_i)
										linkR(pp3_i, pp4_i)
										linkT(pp2_i, pp4_i)

										faces.append([pp3_i, pp4_i, pp2_i])
										faces.append([pp3_i, c_pp1_i, pp2_i])
										t1 = grid_vertices_grid[i + 1][j - k].getPosition()
										t2 = grid_vertices_grid[i + 1][j].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

										while(t1[0] > r_p2[0]):
											t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
											t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

											linkR(t1_i, t2_i)
											linkT(t2_i, c_pp1_i)
											faces.append([t1_i, t2_i, c_pp1_i])

											t2 = copy.deepcopy(t1)
											k = k + 1

											t1 = grid_vertices_grid[i + 1][j - k].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
										linkR(c_pp2_i, t2_i)
										faces.append([c_pp1_i, t2_i, c_pp2_i])

						# 交线尾部闭合
						elif(border_flag == -1):
							
							p1_i = vertices_hash[p2str([p1[0], p1[1], p1[2][0]])]
							p2_i = vertices_hash[p2str([p2[0], p2[1], p2[2][0]])]
							p3_i = vertices_hash[p2str([p3[0], p3[1], p3[2][0]])]
							p4_i = vertices_hash[p2str([p4[0], p4[1], p4[2][0]])]
							pre_p1_i = vertices_hash[p2str([pre_p1[0], pre_p1[1], pre_p1[2][0]])]
							pre_p2_i = vertices_hash[p2str([pre_p2[0], pre_p2[1], pre_p2[2][0]])]
						
							# 无重值点左右交线拼接
							if(p1[0] < l_p[0] and p3[0] < l_p2[0] and p2[0] > r_p[0] and p4[0] > r_p2[0]):
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
								l_c_pp2 = [l_p[0], l_p[1], 0.01]

								o_c_pp = [l_p2[0], l_p2[1], l_p2[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
								r_c_pp2 = [r_p[0], r_p[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

								linkR(p1_i, l_c_pp2_i)
								linkR(p3_i, o_c_pp_i)

								linkR(r_c_pp2_i, p2_i)
								linkR(o_c_pp_i, p4_i)


								faces.append([p1_i, p3_i, o_c_pp_i])
								faces.append([p1_i, o_c_pp_i, l_c_pp2_i])

								faces.append([p2_i, p4_i, r_c_pp2_i])
								faces.append([r_c_pp2_i, o_c_pp_i, p4_i])

								# 多层部分
								faces.append([l_c_pp1_i, r_c_pp2_i, o_c_pp_i])
								faces.append([l_c_pp2_i, r_c_pp1_i, o_c_pp_i])

								faces.append([o_c_pp_i, p3_i, p4_i])

							
							# 交线在两个格子,重复点位p1
							if(p1[0] > l_p[0] and pre_p1[0] < l_p[0] and p1[0] < r_p[0] and p2[0] > r_p[0] and p3[0] < l_p2[0] and p4[0] > l_p2[0]):
								p1_2_i = vertices_hash[p2str([p1[0], p1[1], p1[2][1]])]
								
								l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
								l_c_pp2 = [l_p[0], l_p[1], 0.01]

								o_c_pp = [l_p2[0], l_p2[1], l_p2[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
								r_c_pp2 = [r_p[0], r_p[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

								linkR(pre_p1_i, l_c_pp2_i)
								linkR(l_c_pp2_i, p1_2_i)
								linkR(l_c_pp1_i, p1_i)
								linkR(p1_2_i, r_c_pp1_i)
								linkR(p1_i, r_c_pp2_i)
								linkR(r_c_pp2_i, p2_i)
								linkR(pre_p2_i, p3_i)
								linkR(p3_i, o_c_pp_i)
								linkR(o_c_pp_i, p4_i)

								faces.append([l_c_pp2_i, p1_2_i, o_c_pp_i])
								faces.append([l_c_pp1_i, p1_i, o_c_pp_i])
								faces.append([r_c_pp1_i, p1_2_i, o_c_pp_i])
								faces.append([r_c_pp2_i, p1_i, o_c_pp_i])
								faces.append([pre_p1_i, pre_p2_i, l_c_pp2_i])
								faces.append([p3_i, pre_p2_i, l_c_pp2_i])
								faces.append([p3_i, o_c_pp_i, l_c_pp2_i])
								faces.append([p4_i, o_c_pp_i, r_c_pp2_i])
								faces.append([p4_i, p2_i, r_c_pp2_i])
								faces.append([o_c_pp_i, p3_i, p4_i])

							# 交线在两个格子,重复点位p1
							if(p3[0] > r_p2[0] and pre_p2[0] < l_p2[0] and pre_p1[0] < l_p[0] and p1[0] > l_p[0] and p1[0] < r_p[0] and p2[0] > r_p[0]):
								p1_2_i = vertices_hash[p2str([p1[0], p1[1], p1[2][1]])]
								
								l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
								l_c_pp2 = [l_p[0], l_p[1], 0.01]

								o_c_pp = [l_p2[0], l_p2[1], l_p2[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
								r_c_pp2 = [r_p[0], r_p[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

								linkR(pre_p1_i, l_c_pp2_i)
								linkR(l_c_pp2_i, p1_2_i)
								linkR(l_c_pp1_i, p1_i)
								linkR(p1_2_i, r_c_pp1_i)
								linkR(p1_i, r_c_pp2_i)
								linkR(r_c_pp2_i, p2_i)
								linkR(pre_p2_i, o_c_pp_i)
								linkR(o_c_pp_i, p3_i)
								linkR(p3_i, p4_i)
								faces.append([l_c_pp2_i, p1_2_i, o_c_pp_i])
								faces.append([l_c_pp1_i, p1_i, o_c_pp_i])
								faces.append([r_c_pp1_i, p1_2_i, o_c_pp_i])
								faces.append([r_c_pp2_i, p1_i, o_c_pp_i])
								faces.append([pre_p1_i, pre_p2_i, l_c_pp2_i])
								faces.append([o_c_pp_i, pre_p2_i, l_c_pp2_i])
								faces.append([p2_i, p4_i, r_c_pp2_i])
								faces.append([p4_i, p3_i, r_c_pp2_i])
								faces.append([o_c_pp_i, p3_i, r_c_pp2_i])
								faces.append([o_c_pp_i, p3_i, pre_p2_i])
							if(p1[0] > l_p[0] and p3[0] < l_p2[0] and p2[0] < r_p[0] and p4[0] > r_p2[0]):
								next_p1 = grid_vertices_grid[i][j + 2].getPosition()
								next_p2 = grid_vertices_grid[i + 1][j + 2].getPosition()
								next_p1_i = vertices_hash[p2str([next_p1[0], next_p1[1], next_p1[2][0]])]
								next_p2_i = vertices_hash[p2str([next_p2[0], next_p2[1], next_p2[2][0]])]
								p1_2_i = vertices_hash[p2str([p1[0], p1[1], p1[2][1]])]
								p2_2_i = vertices_hash[p2str([p2[0], p2[1], p2[2][1]])]
								
								l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
								l_c_pp2 = [l_p[0], l_p[1], 0.01]

								o_c_pp = [l_p2[0], l_p2[1], l_p2[2]]

								l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
								l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]
								o_c_pp_i = vertices_hash[p2str(o_c_pp)]

								r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
								r_c_pp2 = [r_p[0], r_p[1], 0.01]
								r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
								r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
								linkR(l_c_pp2_i, p1_2_i)
								linkR(l_c_pp1_i, p1_i)
								linkR(p1_2_i, p2_2_i)
								linkR(p2_2_i, r_c_pp1_i)
								linkR(p2_i, r_c_pp2_i)
								linkR(p3_i, o_c_pp_i)
								linkR(o_c_pp_i, p4_i)
								linkR(pre_p1_i, l_c_pp2_i)
								linkR(pre_p2_i, p3_i)
								linkR(r_c_pp2_i, next_p1_i)
								faces.append([l_c_pp2_i, p1_2_i, o_c_pp_i])
								faces.append([l_c_pp1_i, p1_i, o_c_pp_i])
								faces.append([r_c_pp1_i, p2_2_i, o_c_pp_i])
								faces.append([r_c_pp2_i, p2_i, o_c_pp_i])
								faces.append([o_c_pp_i, p1_i, p2_i])
								faces.append([o_c_pp_i, p1_2_i, p2_2_i])
								faces.append([pre_p1_i, pre_p2_i, l_c_pp2_i])
								faces.append([pre_p2_i, p3_i, l_c_pp2_i])
								faces.append([o_c_pp_i, p3_i, l_c_pp2_i])

								faces.append([next_p1_i, next_p2_i, r_c_pp2_i])
								faces.append([p4_i, next_p2_i, r_c_pp2_i])
								faces.append([p4_i, o_c_pp_i, r_c_pp2_i])
								faces.append([p4_i, o_c_pp_i, p3_i])
						else:
							if(p2[0] > l_p_min[0] and p1[0] < r_p_max[0]):
								if(q == 5 and TYPES[u] == 4):
									if(len(p1[2]) == 3):
										p1[2].pop(0)
									if(len(p2[2]) == 3):
										p2[2].pop(0)
									if(len(p3[2]) == 3):
										p3[2].pop(0)
									if(len(p4[2]) == 3):
										p4[2].pop(0)
								
							# 交线多边形内的多层
							if(p1[0] > l_p_max[0] and p1[0] < r_p_min[0]):
								if(TYPES[u] == 4 and q != 5):
									if(len(p1[2]) == 3):
										grid_vertices_grid[i][j].getPosition()[2].pop(0)
								if(q == 5):
									l = 2
								else:
									l = len(p1[2])
								for m in range(l):
									if(TYPES[u] == 3 and q != 5):
										if(m == 2):
											grid_vertices_grid[i][j].getPosition()[2].pop(0)
											if(len(grid_vertices_grid[i][j - 1].getPosition()[2]) == 3):
												grid_vertices_grid[i][j - 1].getPosition()[2].pop(0)
											continue
									if(q == 5 and u == 12 and m == 0):
										if(f8_flag):
											f8_l = f8_cross[f8_i - 1][0]
											f8_r = f8_cross[f8_i - 1][1]
											f8_l2 = f8_cross[f8_i][0]
											f8_r2 = f8_cross[f8_i][1]
											if((p2[0] > f8_l[0] and p1[0] < f8_r[0]) or (p4[0] > f8_l2[0] and p3[0] < f8_r2[0])):
												continue
										if(f4_flag):
											f4_l = f4_cross[f4_i - 1][0]
											f4_r = f4_cross[f4_i - 1][1]
											f4_l2 = f4_cross[f4_i][0]
											f4_r2 = f4_cross[f4_i][1]
											if((p2[0] > f4_l[0] and p1[0] < f4_r[0]) or (p4[0] > f4_l2[0] and p3[0] < f4_r2[0])):
												continue
										if(f6_flag):
											f6_l = f6_cross[f6_i - 1][0]
											f6_r = f6_cross[f6_i - 1][1]
											f6_l2 = f6_cross[f6_i][0]
											f6_r2 = f6_cross[f6_i][1]
											if((p2[0] > f6_l[0] and p1[0] < f6_r[0]) or (p4[0] > f6_l2[0] and p3[0] < f6_r2[0])):
												continue
								
									pp1 = [p1[0], p1[1], p1[2][m]]
									pp3 = [p3[0], p3[1], p3[2][m]]
									# 中间只有一条线为多层
									only_one = False
									if(p1[0] > l_p_max[0] and p2[0] < r_p_min[0]):
										pp2 = [p2[0], p2[1], p2[2][m]]
										pp4 = [p4[0], p4[1], p4[2][m]]
									else:
										pp2 = pp1
										pp4 = pp3	
										only_one = True

									pp1_i = vertices_hash[p2str(pp1)]
									pp2_i = vertices_hash[p2str(pp2)]
									pp3_i = vertices_hash[p2str(pp3)]
									pp4_i = vertices_hash[p2str(pp4)]

									if(only_one):
										linkT(pp1_i, pp3_i)

									else:
										linkR(pp1_i, pp2_i)
										linkR(pp3_i, pp4_i)
										linkT(pp1_i, pp3_i)
										linkT(pp2_i, pp4_i)

										faces.append([pp1_i, pp2_i, pp3_i])
										faces.append([pp2_i, pp3_i, pp4_i])

									# 内层多边形与左边交线相连
									if(TYPES[u] == 4 and  m == 0 and v1.getType() == 10 and not (q == 5 and u == 9 and p1[1] > 0.25)):
										continue
									if(TYPES[u] == 4 and  m == 0 and v2.getType() == 10 and not (q == 5 and u == 9 and p1[1] > 0.25)):
										continue
									if(TYPES[u] == 4 and  m == 0 and v3.getType() == 10 and not (q == 5 and u == 9 and p1[1] > 0.25)):
										continue
									if(TYPES[u] == 4 and  m == 0 and v4.getType() == 10 and not (q == 5 and u == 9 and p1[1] > 0.25)):
										continue
									pre_p1 = grid_vertices_grid[i][j - 1].getPosition()
									pre_p2 = grid_vertices_grid[i + 1][j - 1].getPosition()
									pre_p3 = grid_vertices_grid[i][j - 2].getPosition()
									pre_p4 = grid_vertices_grid[i + 1][j - 2].getPosition()
									c_pp1 = [l_p[0], l_p[1], l_p[2]]
									c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										c_pp1[2] = m * 0.01
										c_pp2[2] = m * 0.01
									c_pp1_i = vertices_hash[p2str(c_pp1)]
									c_pp2_i = vertices_hash[p2str(c_pp2)]
									# 交线在一个格子内
									if(pre_p1[0] < l_p[0] and pre_p2[0] < l_p2[0] and p1[0] > l_p[0] and p3[0] > l_p2[0]):
										
										linkR(c_pp1_i, pp1_i)
										linkR(c_pp2_i, pp3_i)

										faces.append([c_pp1_i, pp1_i, pp3_i])
										faces.append([c_pp1_i, c_pp2_i, pp3_i])
							
									# 交线在2个格子以上内
									elif(pre_p2[0] < l_p2[0] and p3[0] > l_p2[0] and p1[0] > l_p[0] and pre_p1[0] > l_p[0]):
										
										k = 1
										pre_pp1 = [pre_p1[0], pre_p1[1], pre_p1[2][m]]

										pre_pp1_i = vertices_hash[p2str(pre_pp1)]

										linkR(c_pp2_i, pp3_i)
										linkR(pre_pp1_i, pp1_i)
										linkT(pp1_i, pp3_i)

										faces.append([pre_pp1_i, pp1_i, pp3_i])
										faces.append([pre_pp1_i, c_pp2_i, pp3_i])
										t1 = grid_vertices_grid[i][j - 1 - k].getPosition()
										t2 = grid_vertices_grid[i][j - 1].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]

										while(t1[0] > l_p[0]):
											t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][m]])]
											t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]

											linkR(t1_i, t2_i)
											faces.append([t1_i, t2_i, c_pp2_i])

											t2 = t1
											k = k + 1

											t1 = grid_vertices_grid[i][j - 1 - k].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]
										linkR(c_pp1_i, t2_i)
										faces.append([c_pp1_i, t2_i, c_pp2_i])
									# 交线在2个格子以上内,反向
									elif(pre_p1[0] < l_p[0] and p1[0] > l_p[0] and p3[0] > l_p2[0] and pre_p2[0] > l_p2[0]):
										k = 1
										pre_pp2_i = vertices_hash[p2str([pre_p2[0], pre_p2[1], pre_p2[2][m]])]

										linkR(c_pp1_i, pp1_i)
										linkR(pre_pp2_i, pp3_i)
										linkT(pp1_i, pp3_i)

										faces.append([pre_pp2_i, pp3_i, pp1_i])
										faces.append([pre_pp2_i, c_pp1_i, pp1_i])
										t1 = grid_vertices_grid[i + 1][j - 1 - k].getPosition()
										t2 = grid_vertices_grid[i + 1][j - 1].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]

										while(t1[0] > l_p2[0]):
											t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][m]])]
											t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]

											linkR(t1_i, t2_i)
											faces.append([t1_i, t2_i, c_pp1_i])

											t2 = t1
											k = k + 1

											t1 = grid_vertices_grid[i + 1][j - 1 - k].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]
										linkR(c_pp2_i, t2_i)
										faces.append([c_pp2_i, t2_i, c_pp1_i])
									# 内层多边形与右边交线相连
									
									if(only_one):
										g = 1
										next_p1 = grid_vertices_grid[i][j + 1].getPosition()
										next_p2 = grid_vertices_grid[i + 1][j + 1].getPosition()

									else:
										g = 2
										next_p1 = grid_vertices_grid[i][j + 2].getPosition()
										next_p2 = grid_vertices_grid[i + 1][j + 2].getPosition()
										
									c_pp1 = [r_p[0], r_p[1], r_p[2]]
									c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										c_pp1[2] = 0.01
										c_pp2[2] = 0.01
									c_pp1_i = vertices_hash[p2str(c_pp1)]
									c_pp2_i = vertices_hash[p2str(c_pp2)]
									# 交线在一个格子内
									if(next_p1[0] > r_p[0] and next_p2[0] > r_p2[0] and pp2[0] < r_p[0] and pp4[0] < r_p2[0]):
										linkR(pp2_i, c_pp1_i)
										linkR(pp4_i, c_pp2_i)

										faces.append([c_pp1_i, pp2_i, pp4_i])
										faces.append([c_pp1_i, c_pp2_i, pp4_i])

									# 交线在2个格子以上内
									elif(pp4[0] < r_p2[0] and next_p2[0] > r_p2[0] and pp2[0] < r_p[0] and next_p1[0] < r_p[0]):
										k = 1
										next_pp1_i = vertices_hash[p2str([next_p1[0], next_p1[1], next_p1[2][m]])]
										if(q == 5 and u == 12):
											if(f8_flag and not only_one):
												f8_l = f8_cross[f8_i - 1][0]
												f8_r = f8_cross[f8_i - 1][1]
												f8_l2 = f8_cross[f8_i][0]
												f8_r2 = f8_cross[f8_i][1]
												if((next_p1[0] > f8_l[0]) or (next_p2[0] > f8_l2[0])):
													continue
											if(f4_flag and not only_one):
												f4_l = f4_cross[f4_i - 1][0]
												f4_r = f4_cross[f4_i - 1][1]
												f4_l2 = f4_cross[f4_i][0]
												f4_r2 = f4_cross[f4_i][1]
												if((next_p1[0] > f4_l[0]) or (next_p2[0] > f4_l2[0])):
													continue
											if(f6_flag and not only_one):
												f6_l = f6_cross[f6_i - 1][0]
												f6_r = f6_cross[f6_i - 1][1]
												f6_l2 = f6_cross[f6_i][0]
												f6_r2 = f6_cross[f6_i][1]
												if((next_p1[0] > f6_l[0]) or (next_p2[0] > f6_l2[0])):
													continue
									
										linkR(pp2_i, next_pp1_i)
										linkR(pp4_i, c_pp2_i)
										linkT(pp2_i, pp4_i)

										faces.append([next_pp1_i, pp2_i, pp4_i])
										faces.append([c_pp2_i, next_pp1_i, pp4_i])
										
										t1 = grid_vertices_grid[i][j + g + k].getPosition()
										t2 = grid_vertices_grid[i][j + g].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]

										while(t1[0] < r_p[0]):
											t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][m]])]
											t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]

											linkR(t2_i, t1_i)
											faces.append([t1_i, t2_i, c_pp2_i])

											t2 = t1
											k = k + 1

											t1 = grid_vertices_grid[i][j + g + k].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]
										linkR(t2_i, c_pp1_i)
										faces.append([c_pp1_i, t2_i, c_pp2_i])
									# 交线在2个格子以上内,反向
									elif(pp2[0] < r_p[0] and next_p1[0] > r_p[0] and pp4[0] < r_p2[0] and next_p2[0] < r_p2[0]):
										k = 1
										next_pp2_i = vertices_hash[p2str([next_p2[0], next_p2[1], next_p2[2][m]])]
										if(q == 5 and u == 12):
											if(f8_flag and not only_one):
												f8_l = f8_cross[f8_i - 1][0]
												f8_r = f8_cross[f8_i - 1][1]
												f8_l2 = f8_cross[f8_i][0]
												f8_r2 = f8_cross[f8_i][1]
												if((next_p1[0] > f8_l[0]) or (next_p2[0] > f8_l2[0])):
													continue
											if(f4_flag and not only_one):
												f4_l = f4_cross[f4_i - 1][0]
												f4_r = f4_cross[f4_i - 1][1]
												f4_l2 = f4_cross[f4_i][0]
												f4_r2 = f4_cross[f4_i][1]
												if((next_p1[0] > f4_l[0]) or (next_p2[0] > f4_l2[0])):
													continue
											if(f6_flag and not only_one):
												f6_l = f6_cross[f6_i - 1][0]
												f6_r = f6_cross[f6_i - 1][1]
												f6_l2 = f6_cross[f6_i][0]
												f6_r2 = f6_cross[f6_i][1]
												if((next_p1[0] > f6_l[0]) or (next_p2[0] > f6_l2[0])):
													continue
									
										linkR(pp4_i, next_pp2_i)
										linkR(pp2_i, c_pp1_i)
										linkT(pp2_i, pp4_i)

										faces.append([next_pp2_i, pp4_i, pp2_i])
										faces.append([c_pp1_i, next_pp2_i, pp2_i])
										t1 = grid_vertices_grid[i + 1][j + g + k].getPosition()
										t2 = grid_vertices_grid[i + 1][j + g].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]
										while(t1[0] < r_p2[0]):
											t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][m]])]
											t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]

											linkR(t2_i, t1_i)
											faces.append([t1_i, t2_i, c_pp1_i])

											t2 = t1
											k = k + 1
											t1 = grid_vertices_grid[i + 1][j + g + k].getPosition()
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][m]])]
										linkR(t2_i, c_pp2_i)
										faces.append([c_pp2_i, t2_i, c_pp1_i])
							
							# 交线多边形内只有一个重值点p1
							elif(p1[0] > l_p[0] and p3[0] < l_p2[0] and p2[0] > r_p[0] and p4[0] > r_p2[0] and p1[0] < r_p[0] and grid_vertices_grid[i][j - 1].getPosition()[0] < l_p[0]):
								pi = p1
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

									pp = [pi[0], pi[1], pi[2][m]]
									pp_i = vertices_hash[p2str(pp)]

									linkR(l_c_pp1_i, pp_i)
									linkR(pp_i, r_c_pp1_i)
									linkR(l_c_pp2_i, r_c_pp2_i)
									faces.append([pp_i, l_c_pp1_i, l_c_pp2_i])
									faces.append([l_c_pp2_i, r_c_pp1_i, r_c_pp2_i])	
									faces.append([pp_i, r_c_pp1_i, l_c_pp2_i])	
							
							# 交线多边形内只有一个重值点p2
							elif(p1[0] < l_p[0] and p3[0] < l_p2[0] and p2[0] < r_p[0] and p4[0] > r_p2[0] and p2[0] > l_p[0] and  grid_vertices_grid[i][j + 2].getPosition()[0] > r_p[0]):
								pi = p2
								
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

									pp = [pi[0], pi[1], pi[2][m]]
									pp_i = vertices_hash[p2str(pp)]

									linkR(l_c_pp1_i, pp_i)
									linkR(pp_i, r_c_pp1_i)
									linkR(l_c_pp2_i, r_c_pp2_i)

									faces.append([pp_i, r_c_pp1_i, r_c_pp2_i])
									faces.append([pp_i, l_c_pp2_i, r_c_pp2_i])	
									faces.append([pp_i, l_c_pp1_i, l_c_pp2_i])	
							
							# 交线多边形内只有一个重值点p3
							elif(p1[0] < l_p[0] and p3[0] > l_p2[0] and p2[0] > r_p[0] and p4[0] > r_p2[0] and p3[0] < r_p2[0] and grid_vertices_grid[i + 1][j - 1].getPosition()[0] < l_p2[0]):
								pi = p3
								
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

									pp = [pi[0], pi[1], pi[2][m]]
									pp_i = vertices_hash[p2str(pp)]

									linkR(l_c_pp2_i, pp_i)
									linkR(pp_i, r_c_pp2_i)
									linkR(l_c_pp1_i, r_c_pp1_i)
									faces.append([pp_i, l_c_pp2_i, l_c_pp1_i])
									faces.append([pp_i, l_c_pp1_i, r_c_pp2_i])	
									faces.append([l_c_pp1_i, r_c_pp1_i, r_c_pp2_i])	
							
							# 交线多边形内只有一个重值点p4
							elif(p1[0] < l_p[0] and p3[0] < l_p2[0] and p2[0] > r_p[0] and p4[0] < r_p2[0] and p4[0] > l_p2[0] and grid_vertices_grid[i + 1][j + 2].getPosition()[0] > r_p2[0]):
								pi = p4
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

									pp = [pi[0], pi[1], pi[2][m]]
									pp_i = vertices_hash[p2str(pp)]

									linkR(l_c_pp2_i, pp_i)
									linkR(pp_i, r_c_pp2_i)
									linkR(l_c_pp1_i, r_c_pp1_i)
									faces.append([pp_i, r_c_pp2_i, r_c_pp1_i])
									faces.append([pp_i, r_c_pp1_i, l_c_pp2_i])	
									faces.append([l_c_pp1_i, l_c_pp2_i, r_c_pp1_i])	
								
							# 交线多边形内只有两个对交线重值点,平行四边形
							elif(p1[0] < l_p[0] and l_p[0] < p2[0] and p3[0] > l_p2[0] and p2[0] < r_p[0] and p4[0] > r_p2[0] and r_p2[0] > p3[0]):
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
									pp2 = [p2[0], p2[1], p2[2][m]]
									pp2_i = vertices_hash[p2str(pp2)]
									pp3 = [p3[0], p3[1], p3[2][m]]
									pp3_i = vertices_hash[p2str(pp3)]


									linkR(l_c_pp2_i, pp3_i)
									linkR(pp3_i, r_c_pp2_i)
									linkR(pp2_i, r_c_pp1_i)
									linkR(l_c_pp1_i, pp2_i)

									faces.append([l_c_pp1_i, l_c_pp2_i, pp3_i])
									faces.append([pp3_i, l_c_pp1_i, pp2_i])
									faces.append([pp3_i, pp2_i, r_c_pp2_i])
									faces.append([r_c_pp2_i, pp2_i, r_c_pp1_i])
							
							# 交线多边形内只有两个对交线重值点,平行四边形
							elif(p1[0] > l_p[0] and p3[0] < l_p2[0] and p2[0] > r_p[0] and p1[0] < r_p[0] and p4[0] < r_p2[0] and l_p2[0] < p4[0]):
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
									pp1 = [p1[0], p1[1], p1[2][m]]
									pp1_i = vertices_hash[p2str(pp1)]
									pp4 = [p4[0], p4[1], p4[2][m]]
									pp4_i = vertices_hash[p2str(pp4)]

									linkR(l_c_pp2_i, pp4_i)
									linkR(pp4_i, r_c_pp2_i)

									linkR(l_c_pp1_i, pp1_i)
									linkR(pp1_i, r_c_pp1_i)
									linkT(pp1_i, pp4_i)

									faces.append([l_c_pp1_i, l_c_pp2_i, pp1_i])
									faces.append([pp1_i, l_c_pp2_i, pp4_i])
									faces.append([pp4_i, pp1_i, r_c_pp2_i])
									faces.append([r_c_pp2_i, pp1_i, r_c_pp1_i])
							
							# 交线多边形内只有1个重值点p4,平行四边形
							elif(p1[0] > l_p[0] and p3[0] < l_p2[0] and p1[0] > r_p[0] and p4[0] < r_p2[0] and l_p2[0] < p4[0] and grid_vertices_grid[i][j - 1].getPosition()[0] < l_p[0] and grid_vertices_grid[i + 1][j + 2].getPosition()[0] > r_p2[0]):
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								grid_vertices_grid[i][j - 1].setLive(p_live)
								grid_vertices_grid[i + 1][j - 1].setLive(p_live)
								grid_vertices_grid[i][j + 2].setLive(p_live)
								grid_vertices_grid[i + 1][j + 2].setLive(p_live)
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
									pp4 = [p4[0], p4[1], p4[2][m]]
									pp4_i = vertices_hash[p2str(pp4)]
									linkR(l_c_pp2_i, pp4_i)
									linkR(pp4_i, r_c_pp2_i)
									linkR(l_c_pp1_i, r_c_pp1_i)

									faces.append([l_c_pp1_i, l_c_pp2_i, r_c_pp1_i])
									faces.append([r_c_pp1_i, l_c_pp2_i, pp4_i])
									faces.append([pp4_i, r_c_pp1_i, r_c_pp2_i])
							
							# 交线多边形内只有1个重值点p1,平行四边形
							elif(p1[0] > l_p[0] and p1[0] < r_p[0] and p2[0] > r_p[0] and p4[0] < l_p2[0] and  grid_vertices_grid[i + 1][j + 2].getPosition()[0] > r_p2[0] and  grid_vertices_grid[i][j - 1].getPosition()[0] < l_p[0]):
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								grid_vertices_grid[i][j - 1].setLive(p_live)
								grid_vertices_grid[i + 1][j - 1].setLive(p_live)
								grid_vertices_grid[i][j + 2].setLive(p_live)
								grid_vertices_grid[i + 1][j + 2].setLive(p_live)
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
									pp1 = [p1[0], p1[1], p1[2][m]]
									pp1_i = vertices_hash[p2str(pp1)]
									linkR(l_c_pp1_i, pp1_i)
									linkR(pp1_i, r_c_pp1_i)
									linkR(l_c_pp2_i, r_c_pp2_i)

									faces.append([l_c_pp1_i, l_c_pp2_i, pp1_i])
									faces.append([r_c_pp1_i, l_c_pp2_i, pp1_i])
									faces.append([l_c_pp2_i, r_c_pp1_i, r_c_pp2_i])
							
							# 交线多边形内只有1个重值点p3,平行四边形
							elif(p3[0] > l_p2[0] and p3[0] < r_p2[0] and p4[0] > r_p2[0] and p2[0] < l_p[0] and  grid_vertices_grid[i][j + 2].getPosition()[0] > r_p[0] and  grid_vertices_grid[i + 1][j - 1].getPosition()[0] < l_p2[0]):
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								grid_vertices_grid[i][j - 1].setLive(p_live)
								grid_vertices_grid[i + 1][j - 1].setLive(p_live)
								grid_vertices_grid[i][j + 2].setLive(p_live)
								grid_vertices_grid[i + 1][j + 2].setLive(p_live)
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
									pp3 = [p3[0], p3[1], p3[2][m]]
									pp3_i = vertices_hash[p2str(pp3)]

									linkR(l_c_pp2_i, pp3_i)
									linkR(pp3_i, r_c_pp2_i)

									linkR(l_c_pp1_i, r_c_pp1_i)

									faces.append([l_c_pp1_i, l_c_pp2_i, pp3_i])
									faces.append([r_c_pp1_i, l_c_pp1_i, pp3_i])
									faces.append([pp3_i, r_c_pp1_i, r_c_pp2_i])
						
							# 交线多边形内只有1个重值点p2,平行四边形
							elif(p3[0] > r_p2[0] and p1[0] < l_p[0] and p2[0] > l_p[0] and p2[0] < r_p[0] and grid_vertices_grid[i][j + 2].getPosition()[0] > r_p[0] and  grid_vertices_grid[i + 1][j - 1].getPosition()[0] < l_p2[0]):
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								grid_vertices_grid[i][j - 1].setLive(p_live)
								grid_vertices_grid[i + 1][j - 1].setLive(p_live)
								grid_vertices_grid[i][j + 2].setLive(p_live)
								grid_vertices_grid[i + 1][j + 2].setLive(p_live)
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
									pp2 = [p2[0], p2[1], p2[2][m]]
									pp2_i = vertices_hash[p2str(pp2)]
									linkR(l_c_pp1_i, pp2_i)
									linkR(pp2_i, r_c_pp1_i)
									linkR(l_c_pp2_i, r_c_pp2_i)

									faces.append([l_c_pp1_i, l_c_pp2_i, pp2_i])
									faces.append([r_c_pp2_i, l_c_pp2_i, pp2_i])
									faces.append([pp2_i, r_c_pp1_i, r_c_pp2_i])
							
							# 交线多边形内只有两个同边重值点，梯形
							elif(p1[0] > l_p[0] and p2[0] < r_p[0] and p3[0] < l_p2[0] and p4[0] > r_p2[0]):
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
									pp1 = [p1[0], p1[1], p1[2][m]]
									pp1_i = vertices_hash[p2str(pp1)]
									pp2 = [p2[0], p2[1], p2[2][m]]
									pp2_i = vertices_hash[p2str(pp2)]
									linkR(l_c_pp1_i, pp1_i)
									linkR(pp1_i, pp2_i)
									linkR(pp2_i, r_c_pp1_i)
									linkR(l_c_pp2_i, r_c_pp2_i)
									faces.append([l_c_pp1_i, pp1_i, l_c_pp2_i])
									faces.append([r_c_pp2_i, pp1_i, l_c_pp2_i])
									faces.append([pp2_i, pp1_i, r_c_pp2_i])
									faces.append([pp2_i, r_c_pp1_i, r_c_pp2_i])
								
							# 交线多边形内只有两个同边重值点，倒梯形
							elif(p1[0] < l_p[0] and p2[0] > r_p[0] and p3[0] > l_p2[0] and p4[0] < r_p2[0]):
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]
									pp3 = [p3[0], p3[1], p3[2][m]]
									pp3_i = vertices_hash[p2str(pp3)]
									pp4 = [p4[0], p4[1], p4[2][m]]
									pp4_i = vertices_hash[p2str(pp4)]
									linkR(l_c_pp2_i, pp3_i)
									linkR(pp3_i, pp4_i)
									linkR(pp4_i, r_c_pp2_i)
									linkR(l_c_pp1_i, r_c_pp1_i)
									linkT(l_c_pp1_i, pp3_i)
									linkT(r_c_pp2_i, pp4_i)
									faces.append([l_c_pp2_i, pp3_i, l_c_pp1_i])
									faces.append([r_c_pp1_i, pp3_i, l_c_pp1_i])
									faces.append([pp3_i, pp4_i, r_c_pp1_i])
									faces.append([pp4_i, r_c_pp2_i, r_c_pp1_i])
						
							# 无重值点左右交线拼接,垂直
							elif(p1[0] < l_p[0] and p3[0] < l_p2[0] and p2[0] > r_p[0] and p4[0] > r_p2[0]):
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

									if(p1[0] < l_p[0] and p3[0] < l_p2[0] and p2[0] > r_p[0] and p4[0] > r_p2[0]):
										linkR(l_c_pp1_i, r_c_pp1_i)
										linkR(l_c_pp2_i, r_c_pp2_i)

										faces.append([r_c_pp1_i, l_c_pp1_i, l_c_pp2_i])
										faces.append([l_c_pp2_i, r_c_pp1_i, r_c_pp2_i])
							# 两个格子无重复点,向右平行四边形
							elif(p3[0] < l_p2[0] and p4[0] > r_p2[0] and p1[0] > r_p[0] and grid_vertices_grid[i][j - 1].getPosition()[0] < l_p[0]):
								if(u != 13):
									e = u
								pre_p1 = grid_vertices_grid[i][j - 1].getPosition()
								pre_p2 = grid_vertices_grid[i + 1][j - 1].getPosition()
								pre_pp1 = [pre_p1[0], pre_p1[1], pre_p1[2][0]]
								pre_pp1_i = vertices_hash[p2str(pre_pp1)]
								pre_pp2 = [pre_p2[0], pre_p2[1], pre_p2[2][0]]
								pre_pp2_i = vertices_hash[p2str(pre_pp2)]
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								grid_vertices_grid[i][j - 1].setLive(p_live)
								grid_vertices_grid[i + 1][j - 1].setLive(p_live)
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

									linkR(l_c_pp1_i, r_c_pp1_i)
									linkR(l_c_pp2_i, r_c_pp2_i)

									faces.append([r_c_pp1_i, l_c_pp1_i, l_c_pp2_i])
									faces.append([l_c_pp2_i, r_c_pp1_i, r_c_pp2_i])
							# 两个格子无重复点,向左平行四边形
							elif(p1[0] < l_p[0] and p2[0] > r_p[0] and p3[0] > r_p2[0] and grid_vertices_grid[i + 1][j - 1].getPosition()[0] < l_p2[0]):
								if(u != 13):
									e = u
								pre_p1 = grid_vertices_grid[i][j - 1].getPosition()
								pre_p2 = grid_vertices_grid[i + 1][j - 1].getPosition()
								pre_pp1 = [pre_p1[0], pre_p1[1], pre_p1[2][0]]
								pre_pp1_i = vertices_hash[p2str(pre_pp1)]
								pre_pp2 = [pre_p2[0], pre_p2[1], pre_p2[2][0]]
								pre_pp2_i = vertices_hash[p2str(pre_pp2)]
								v1.setLive(p_live)
								v2.setLive(p_live)
								v3.setLive(p_live)
								v4.setLive(p_live)
								grid_vertices_grid[i][j - 1].setLive(p_live)
								grid_vertices_grid[i + 1][j - 1].setLive(p_live)
								for m in range(2):
									l_c_pp1 = [l_p[0], l_p[1], l_p[2]]
									l_c_pp2 = [l_p2[0], l_p2[1], l_p2[2]]
									if(m != 0):
										l_c_pp1[2] = m * 0.01
										l_c_pp2[2] = m * 0.01
									l_c_pp1_i = vertices_hash[p2str(l_c_pp1)]
									l_c_pp2_i = vertices_hash[p2str(l_c_pp2)]

									r_c_pp1 = [r_p[0], r_p[1], r_p[2]]
									r_c_pp2 = [r_p2[0], r_p2[1], r_p2[2]]
									if(m == 0):
										r_c_pp1[2] = 0.01
										r_c_pp2[2] = 0.01
									r_c_pp1_i = vertices_hash[p2str(r_c_pp1)]
									r_c_pp2_i = vertices_hash[p2str(r_c_pp2)]

									linkR(l_c_pp1_i, r_c_pp1_i)
									linkR(l_c_pp2_i, r_c_pp2_i)

									faces.append([r_c_pp1_i, l_c_pp1_i, l_c_pp2_i])
									faces.append([l_c_pp2_i, r_c_pp1_i, r_c_pp2_i])
								
							#网格单层与左交线拼接处
							if(p1[0] < l_p_min[0] and p2[0] > l_p_min[0]):
								if(u == 5):
									if(p1[0] == 0.41 and abs(p1[1] - 0.841) < 0.001):
										e = 0
								if(TYPES[u] == 4 and  v1.getType() == 10  and not (q == 5 and u == 9 and p1[1] > 0.25)):
										continue
								pp1 = [p1[0], p1[1], p1[2][0]]
								pp2 = [p2[0], p2[1], p2[2][0]]
								pp3 = [p3[0], p3[1], p3[2][0]]
								pp4 = [p4[0], p4[1], p4[2][0]]
								pp1_i = vertices_hash[p2str(pp1)]
								pp2_i = vertices_hash[p2str(pp2)]
								pp3_i = vertices_hash[p2str(pp3)]
								pp4_i = vertices_hash[p2str(pp4)]
								
								c_pp1 = [l_p[0], l_p[1], 0.01]
								c_pp2 = [l_p2[0], l_p2[1], 0.01]
								
								c_pp1_i = vertices_hash[p2str(c_pp1)]
								c_pp2_i = vertices_hash[p2str(c_pp2)]


								next_p1 = grid_vertices_grid[i][j + 2].getPosition()
								next_p2 = grid_vertices_grid[i + 1][j + 2].getPosition()
								pre_p1 = grid_vertices_grid[i][j - 1].getPosition()
								pre_p2 = grid_vertices_grid[i + 1][j - 1].getPosition()
								# 交线在一个格子内
								if(p2[0] > l_p_max[0] and p4[0] > l_p_max[0]):
									linkR(pp1_i, c_pp1_i)
									linkR(pp3_i, c_pp2_i)
									linkT(c_pp1_i, c_pp2_i)

									faces.append([pp1_i, pp3_i, c_pp1_i])
									faces.append([c_pp1_i, c_pp2_i, pp3_i])
								# 交线在两个格子以上
								elif(p1[0] < l_p[0] and p2[0] > l_p[0] and p4[0] < l_p2[0]):
									k = 1
									linkR(pp1_i, c_pp1_i)
									linkR(pp3_i, pp4_i)
									linkT(pp1_i, pp3_i)
									
									faces.append([pp1_i, pp3_i, pp4_i])
									faces.append([pp1_i, c_pp1_i, pp4_i])
									t1 = grid_vertices_grid[i + 1][j + k + 1].getPosition()
									t2 = grid_vertices_grid[i + 1][j + 1].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

									while(t1[0] < l_p2[0]):
										t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

										linkR(t2_i, t1_i)
										faces.append([t1_i, t2_i, c_pp1_i])

										t2 = copy.deepcopy(t1)
										k = k + 1

										t1 = grid_vertices_grid[i + 1][j + k].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
									linkR(t2_i, c_pp2_i)
									faces.append([c_pp1_i, t2_i, c_pp2_i])
								# 交线在两个格子以上反
								elif(p3[0] < l_p2[0] and p4[0] > l_p2[0] and p2[0] < l_p[0]):
									k = 1
									linkR(pp3_i, c_pp2_i)
									linkR(pp1_i, pp2_i)
									linkT(pp1_i, pp3_i)
									faces.append([pp1_i, pp2_i, pp3_i])
									faces.append([pp2_i, c_pp2_i, pp3_i])
									t1 = grid_vertices_grid[i][j + k + 1].getPosition()
									t2 = grid_vertices_grid[i][j + 1].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

									while(t1[0] < l_p[0]):
										t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

										linkR(t2_i, t1_i)
										linkT(t2_i, c_pp2_i)
										faces.append([t1_i, t2_i, c_pp2_i])

										t2 = copy.deepcopy(t1)
										k = k + 1

										t1 = grid_vertices_grid[i][j + k].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
									linkR(t2_i, c_pp1_i)
									faces.append([c_pp1_i, t2_i, c_pp2_i])
							# 网格单层与右交线拼接处
							if(p1[0] < r_p_max[0] and p2[0] > r_p_max[0]):
								
								if(q == 5 and u == 12):
									if(f8_flag):
										f8_l = f8_cross[f8_i - 1][0]
										f8_r = f8_cross[f8_i - 1][1]
										f8_l2 = f8_cross[f8_i][0]
										f8_r2 = f8_cross[f8_i][1]
										if(p2[0] >= f8_l[0] and p1[0] <= f8_r[0] or (p4[0] >= f8_l2[0] and p3[0] <= f8_r2[0])):
											continue
									if(f4_flag):
										f4_l = f4_cross[f4_i - 1][0]
										f4_r = f4_cross[f4_i - 1][1]
										f4_l2 = f4_cross[f4_i][0]
										f4_r2 = f4_cross[f4_i][1]
										if(p2[0] >= f4_l[0] and p1[0] <= f4_r[0] or (p4[0] >= f4_l2[0] and p3[0] <= f4_r2[0])):
											continue
									if(f6_flag):
										f6_l = f6_cross[f6_i - 1][0]
										f6_r = f6_cross[f6_i - 1][1]
										f6_l2 = f6_cross[f6_i][0]
										f6_r2 = f6_cross[f6_i][1]
										if(p2[0] >= f6_l[0] and p1[0] <= f6_r[0] or (p4[0] >= f6_l2[0] and p3[0] <= f6_r2[0])):
											continue
							
								if(TYPES[u] == 4 and  v1.getType() == 10 and not (q == 5 and u == 9 and p1[1] > 0.25)):
										continue
								if(len(p2[2]) == 2):
									continue
								if(len(p4[2]) == 2): 
									continue
								if(TYPES[u] == -1):
									continue

								pp1 = [p1[0], p1[1], p1[2][0]]
								pp2 = [p2[0], p2[1], p2[2][0]]
								pp3 = [p3[0], p3[1], p3[2][0]]
								pp4 = [p4[0], p4[1], p4[2][0]]
								pp1_i = vertices_hash[p2str(pp1)]
								pp2_i = vertices_hash[p2str(pp2)]
								pp3_i = vertices_hash[p2str(pp3)]
								pp4_i = vertices_hash[p2str(pp4)]
								pre_p1 = grid_vertices_grid[i][j - 1].getPosition()
								pre_p2 = grid_vertices_grid[i + 1][j - 1].getPosition()
								pre_p3 = grid_vertices_grid[i][j - 2].getPosition()
								pre_p4 = grid_vertices_grid[i + 1][j - 2].getPosition()
								dis_min = 9999
								
								c_pp1 = [r_p[0], r_p[1], 0.01]
								c_pp2 = [r_p2[0], r_p2[1], 0.01]
								
								c_pp1_i = vertices_hash[p2str(c_pp1)]
								c_pp2_i = vertices_hash[p2str(c_pp2)]

								# 交线在一个格子内
								if(p2[0] > r_p[0] and p4[0] > r_p2[0] and p3[0] < r_p2[0] and p1[0] < r_p[0]):
									linkR(c_pp1_i, pp2_i)
									linkR(c_pp2_i, pp4_i)
									linkT(c_pp1_i, c_pp2_i)

									faces.append([pp2_i, pp4_i, c_pp2_i])
									faces.append([c_pp1_i, c_pp2_i, pp2_i])
								# 交线在两个格子以上
								elif(p3[0] < r_p2[0] and p4[0] > r_p2[0] and p1[0] > r_p[0]):
									k = 1
									linkR(c_pp2_i, pp4_i)
									linkR(pp1_i, pp2_i)
									linkT(pp2_i, pp4_i)
									faces.append([pp1_i, pp2_i, pp4_i])
									faces.append([pp1_i, c_pp2_i, pp4_i])
									t1 = grid_vertices_grid[i][j - k].getPosition()
									t2 = grid_vertices_grid[i][j].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

									while(t1[0] > r_p[0]):
										t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

										linkR(t1_i, t2_i)
										linkT(t2_i, c_pp2_i)

										faces.append([t1_i, t2_i, c_pp2_i])

										t2 = copy.deepcopy(t1)
										k = k + 1

										t1 = grid_vertices_grid[i][j - k].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
									linkR(c_pp1_i, t2_i)
									faces.append([c_pp1_i, t2_i, c_pp2_i])
								# 交线在两个格子以上反
								elif(p1[0] < r_p[0] and p2[0] > r_p[0] and p3[0] > r_p2[0]):
									k = 1
									linkR(c_pp1_i, pp2_i)
									linkR(pp3_i, pp4_i)
									linkT(pp2_i, pp4_i)
									
									faces.append([pp3_i, pp4_i, pp2_i])
									faces.append([pp3_i, c_pp1_i, pp2_i])
									t1 = grid_vertices_grid[i + 1][j - k].getPosition()
									t2 = grid_vertices_grid[i + 1][j].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

									while(t1[0] > r_p2[0]):
										t1_i = vertices_hash[p2str([t1[0], t1[1], t1[2][0]])]
										t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]

										linkR(t1_i, t2_i)
										linkT(t2_i, c_pp1_i)
										faces.append([t1_i, t2_i, c_pp1_i])

										t2 = copy.deepcopy(t1)
										k = k + 1

										t1 = grid_vertices_grid[i + 1][j - k].getPosition()
									t2_i = vertices_hash[p2str([t2[0], t2[1], t2[2][0]])]
									linkR(c_pp2_i, t2_i)
									faces.append([c_pp1_i, t2_i, c_pp2_i])
						
		#单层
		left_border = borders_faults[0]
		right_border = borders_faults[0]
		for i in range(len(grid_vertices_grid) - 1):
			for j in range(len(grid_vertices_grid[i]) - 1):
				v1 = grid_vertices_grid[i][j]
				v2 = grid_vertices_grid[i][j + 1]
				v3 = grid_vertices_grid[i + 1][j]
				v4 = grid_vertices_grid[i + 1][j + 1]

				p1 = v1.getPosition()
				p2 = v2.getPosition()
				p3 = v3.getPosition()
				p4 = v4.getPosition()
				
				if(p1[0] >= right_border[i][0]):
					continue
				if(p4[0] >= right_border[i][0]):
					continue
				ps = [p1, p2, p3, p4]
				if(v1.getLive() == v2.getLive() and v2.getLive() == v3.getLive() and v3.getLive() == v4.getLive() and v4.getLive() == v1.getLive() and v1.getLive() == 100):
					p1_i = vertices_hash[p2str([p1[0], p1[1], p1[2][0]])]
					p2_i = vertices_hash[p2str([p2[0], p2[1], p2[2][0]])]
					p3_i = vertices_hash[p2str([p3[0], p3[1], p3[2][0]])]
					p4_i = vertices_hash[p2str([p4[0], p4[1], p4[2][0]])]
					linkR(p1_i, p2_i)
					linkR(p3_i, p4_i)
					linkT(p1_i, p3_i)
					linkT(p2_i, p4_i)

					faces.append([p1_i, p2_i, p3_i])
					faces.append([p2_i, p3_i, p4_i])
				elif(v1.getType() + v2.getType() + v3.getType() + v4.getType() == 0 and not(v1.getLive() == v2.getLive() and v2.getLive() == v3.getLive() and v3.getLive() == v4.getLive() and v4.getLive() == v1.getLive() and v1.getLive() > 2) ):
					p1_i = vertices_hash[p2str([p1[0], p1[1], p1[2][0]])]
					p2_i = vertices_hash[p2str([p2[0], p2[1], p2[2][0]])]
					p3_i = vertices_hash[p2str([p3[0], p3[1], p3[2][0]])]
					p4_i = vertices_hash[p2str([p4[0], p4[1], p4[2][0]])]
					linkR(p1_i, p2_i)
					linkR(p3_i, p4_i)
					linkT(p1_i, p3_i)
					linkT(p2_i, p4_i)

					faces.append([p1_i, p2_i, p3_i])
					faces.append([p2_i, p3_i, p4_i])	
				
		# plt.figure('一层')
		# plt.scatter(x, y, s=1, c='r')

		# plt.scatter(x2, y2, s=1, c='b')
		# plt.scatter(x3, y3, s=1, c='black')
		# for i in range(len(crossPoints_grid_pairs)):
		# 	if(TYPES[i] == 0):
		# 		continue
			# top_line = np.array(crossPoints_grid_pairs[i][0]).T
			# bottom_line = np.array(crossPoints_grid_pairs[i][1]).T
			# plt.scatter(top_line[0], top_line[1], s=1, c='g')
			# plt.scatter(bottom_line[0], bottom_line[1], s=1, c='y')
		# plt.show()
		
		# 偏微分插值，分为两部分，第一部分，原始数据、交线为控制点，对网格进行插值。第二部分，交线为控制点，对数据进行光滑调整。
		for t in range(smooth_times):
			print(catalog.layers[q]+":"+str(t))
			res_v_z = []
			for i in range(len(Vertexs)):
				v = Vertexs[i]
				branch = v.getBranch()
				v_z = v.getZ()
				if(branch[0] == -1):
					top_z = v.getZ()
				else:
					top_z = Vertexs[branch[0]].getZ()
				if(branch[1] == -1):
					bottom_z = v.getZ()
				else:
					bottom_z = Vertexs[branch[1]].getZ()
				if(branch[2] == -1):
					left_z = v.getZ()
				else:
					left_z = Vertexs[branch[2]].getZ()
				if(branch[3] == -1):
					right_z = v.getZ()
				else:
					right_z = Vertexs[branch[3]].getZ()
				if(t == 20 and v.getType() != 3):
					v.setControl(0)
					continue
				if(v.getControl() == 1):
					res_v_z.append(v_z)
				else:
					laplace = (top_z + left_z + right_z + bottom_z - 4 * v_z) / 4
					new_v_z = v_z + laplace
					res_v_z.append(new_v_z)
			if(t != 20):
				for i in range(len(Vertexs)):
					Vertexs[i].setZ(res_v_z[i])
		face_hash = {}

		# 去除没有用到的三角
		for i in range(len(faces)):
			face_hash[faces[i][0]] = 1
			face_hash[faces[i][1]] = 1
			face_hash[faces[i][2]] = 1
		ind = -1
		for k in face_hash.keys():
			ind = ind + 1
			face_hash[k] = ind
		fs = []
		for i in range(len(Vertexs)):
			fs.append(Vertexs[i].getPosition())
		vertices = []
		for k in face_hash.keys():
			vertices.append(fs[k])
		faces_new = []
		for i in range(len(faces)):
			faces_new.append([face_hash[faces[i][0]], face_hash[faces[i][1]], face_hash[faces[i][2]]])
		output = []
		faces = faces_new

		# 顶点
		for i in range(len(vertices)):
			output.append(['v', 
			str(round(vertices[i][1] * (maxY - minY) + minY, 3)), 
			str(round(vertices[i][0] * (maxX - minX) + minX, 3)), 
			str(round(-vertices[i][2] * (maxZ - minZ) + minZ, 3))])
		# 三角面
		for i in range(len(faces)):
			output.append(['f', 
			str(faces[i][0] + 1), 
			str(faces[i][1] + 1), 
			str(faces[i][2] + 1)])
		np.savetxt('obj/' + catalog.layers[q] + ".obj", np.array(output), delimiter=" ",  fmt='%s')