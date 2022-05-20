import numpy as np
def isnumber(aString):
  	try:
  	  	float(aString)
  	  	return True
  	except:
  	  	return False
# 归一化
def normalize(x, min, max):
  	arr = np.array(x)
  	for i in range(len(arr)):
  	  	x[i] = (x[i] - min) / (max - min)
  	return x

# 交线光滑
def crossLineSmooth(top, bottom, s = 0):
	mid_list_x = []
	mid_list_y = top[1]

	distance_list_x = []
	for i in range(len(top[0])):
		x1 = top[0][i]
		x2 = bottom[0][i]

		mid_list_x.append((x1 + x2) / 2)
		distance_list_x.append(abs(x1 - x2) / 2)
	res_dis = inter(mid_list_y, distance_list_x)
	res_mid = inter(mid_list_y, mid_list_x)

	for i in range(s):
		res_dis = inter(mid_list_y, res_dis)
		res_mid = inter(mid_list_y, res_mid)

	top_res_x = []
	bottom_res_x = []
	for i in range(len(mid_list_x)):
		top_res_x.append(res_mid[i] + res_dis[i] )
		bottom_res_x.append(res_mid[i] - res_dis[i])
	top[0] = top_res_x
	bottom[0] = bottom_res_x
	return [top, bottom]
def inter(x, y):
	res = []
	for i in range(len(x) - 1):
		if(i == 0):
			inter_temp = y[i]
		else:
			inter_temp = y[i] + (y[i - 1] - y[i] * 2 + y[i + 1]) / 4
		res.append(inter_temp)
	res.append(y[len(x) - 1])
	return res
# 对交线按照测线采样
def crossLineTraceSampling(crossPointsData, traceData):
	crossPointsData_trace = []
	flag = False
	for j in range(len(crossPointsData)):
		crossPointsData_trace_temp = []
		for k in range(len(crossPointsData[j])):
			if(k == 0):
				crossPointsData_trace_temp.append(crossPointsData[j][k])
				flag = True
			if(k == len(crossPointsData[j]) - 1):
				crossPointsData_trace_temp.append(crossPointsData[j][k])
				flag = True
			for i in range(len(traceData)):
				yi = traceData[i][0][1]
				if(crossPointsData[j][k][1] == yi):
					if((k == 0 and flag) or (k == len(crossPointsData[j]) - 1 and flag)):
						flag = False
						break
					crossPointsData_trace_temp.append(crossPointsData[j][k])
			
			
		crossPointsData_trace.append(crossPointsData_trace_temp)
	return crossPointsData_trace
# 对交线按照网格采样
def crossLineGridSampling(crossPointsData, grid_y):
	top = crossPointsData
	crossPoints_grid = []
	for i in range(len(grid_y)):
		y1 = grid_y[i]
		min = 9999
		min_index = -1
		if(y1 > top[1][0] and y1 < top[1][len(top[1]) - 1]):
			for j in range(len(top[1])):
				y2 = top[1][j]
				if(abs(y2 - y1) < min):
					min = abs(y2 - y1)
					min_index = j
			crossPoints_grid.append([top[0][min_index], grid_y[i], top[2][min_index]])
	crossPoints_grid.insert(0, [top[0][0], top[1][0], top[2][0]])
	crossPoints_grid.append([top[0][len(top[1]) - 1], top[1][len(top[1]) - 1], top[2][len(top[1]) - 1]])
	return crossPoints_grid
# 一维点云转换为道集数据,data 为position
def dataToTraceData(data):
	trace_y = data[0][1]
	trace_temp = []
	sample_trace_data = []
	for i in range(len(data)):
		xi = data[i][0]
		yi = data[i][1]
		zi = data[i][2]
	
		if(yi != trace_y):
			trace_y = yi
			sample_trace_data.append(trace_temp)
			trace_temp = []
			trace_temp.append([xi, yi, zi])
		else:
			trace_temp.append([xi, yi, zi])
	sample_trace_data.append(trace_temp)
	return sample_trace_data

def findVertex(vertices, index):
	for i in range(len(vertices)):
		if(vertices[i].getIndex() == index):
			return vertices[i]
		else:
			return None
def findVertexLive(vertices):
	v = []
	for i in range(len(vertices)):
		if(vertices[i].getLive() == 1):
			v.append(vertices[i])
	return v	

	