'''
功能：求断层层位交线
'''
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from Catalog import Catalog
def printNomal(str):
	print("" + str +"") 
def printError(str):
	print("" + str +"  ×") 
def printSuccess(str):
	print("" + str +"  √") 
# 层位数据分片
def line_segmentation(line,threshold_z,threshold_x):
	start = 0
	seg_cell = []
	seg_mid = []
	res = []
	m = {}
	for i in range(len(line) - 1):
		if(abs(line[i][2] - line[i + 1][2]) > threshold_z or abs(line[i][1] - line[i + 1][1]) > threshold_x):
			seg_cell.append(line[start: i + 1])
			mid = (line[start][1] + line[i][1]) / 2
			seg_mid.append(mid)
			start = i + 1
		if(i == len(line) - 2):
			seg_cell.append(line[start: i + 2])
			mid = (line[start][1] + line[i][1]) / 2
			seg_mid.append(mid)
	for i in range(len(seg_mid)):
		m[seg_mid[i]] = i
	# 排序
	seg_mid.sort()
	for i in range(len(seg_cell)):
		ind = m[seg_mid[i]]
		res.append(seg_cell[ind])
	return res

# 两条线段求交点
def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
	res = []
	px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
	py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
	if(px >= x1 and px <= x2):
		res = [px, py]
	return res
# 求层位面与断层面的交点
def findCross(h, f):
	res = []
	for i in range(len(f)):
		if(len(f[i]) == 0):
			continue
		else:
			for m in range(len(h[i]) - 1):
				for n in range(len(f[i]) - 1):
					p1 = f[i][n]
					p2 = f[i][n + 1]
					p3 = h[i][m]
					p4 = h[i][m + 1]
					if(p1[1] == p3[1]):
						a = (p1[2] - p3[2]) * (p2[2] - p4[2])
						if(a < 0):
							p = findIntersection(p1[1], p1[2], p2[1], p2[2], p3[1], p3[2], p4[1], p4[2])
							
							if(len(p) != 0):
								res.append([p3[0], p[0], p[1]])
						if(a == 0):
							if(p1[2] - p3[2] == 0):
								res.append(p3)
							else:
								res.append(p4)
							n = n + 1
	return res

# 层位数据预处理, 删除超出交点的多余层位数据
def layerPreprocess(h, f, top, bottom):
	res = []
	k = 0
	for i in range(len(f)):
		if(len(f[i]) == 0):
			res.append(h[i])
		else:
			if(k >= len(top)):
				res.append(h[i])
				continue
			if(f[i][0][0] != top[k][0]):
				res.append(h[i])
				continue
			t_p = top[k]
			b_p = bottom[k]
			l_p = []
			r_p = []
			if(t_p[1] < b_p[1]):
				l_p = t_p
				r_p = b_p
			else:
				l_p = b_p
				r_p = t_p
			k = k + 1
			temp = []
			temp2 = []
			for j in range(len(h[i])):
				h_p = h[i][j]
				if(h_p[1] >= r_p[1] and abs(h_p[2] - r_p[2]) < 100):
					break
				else:
					temp.append(h_p)

			for j in range(len(h[i])):
				h_p = h[i][-1 - j]
				
				if(h_p[1] <= l_p[1] and abs(h_p[2] - l_p[2]) < 100):
					break
				else:
					temp2.append(h_p)

			temp2.reverse()
			res.append(temp + temp2)
	return res

# 求取层位面与断层面的交线
def Intersction(layerName, faultName, listNum):
	th_z = 50   # 分段阈值--相邻点的z值差
	th_x = 5   # 分段阈值--相邻点的水平距离
	h = []
	path = './data/预处理/层位/' + layerName + '.txt'
	isExists = os.path.exists(path)
	if(not isExists):
		names='./data/层位/' + layerName + '.txt'
		h = np.loadtxt(names).T
		xline = np.arange(np.min(h[0]), np.max(h[0])  + 16, 16).tolist() # 剖面位置
		h[2] = h[4]
		h[2] = h[2] * - 1
		h = [list(h[0]), list(h[1]), list(h[2])]
	else:
		names='./data/预处理/层位/' + layerName + '.txt'
		h = np.loadtxt(names).T
		xline = np.arange(np.min(h[0]), np.max(h[0])  + 16, 16).tolist() # 剖面位置
		h = [list(h[0]), list(h[1]), list(h[2])]

	
	names='./data/断层/' + faultName + '.txt'
	f = np.loadtxt(names).T
	f[2] = f[4]
	f[2] = f[2] * -1
	f = [list(f[0]), list(f[1]), list(f[2])]

	
	# 分剖面，断层填充
	h_cross = [[] for i in range(len(xline))]
	ind = []
	for i in range(len(xline)):
		try:
			ind = np.argwhere(np.array(h[0]) == xline[i])
			for j in range(len(ind)):
				h_cross[i].append([h[0][int(ind[j])], h[1][int(ind[j])], h[2][int(ind[j])]])
		except:
			ind = []
	f_cross = [[] for i in range(len(xline))]
	for i in range(len(xline)):
		try:
			ind = np.argwhere(np.array(f[0]) == xline[i])
			for j in range(len(ind)):
				f_cross[i].append([f[0][int(ind[j])], f[1][int(ind[j])], f[2][int(ind[j])]])
		except:
			ind = []

	# 层位数据分片
	h_seg = []
	for i in range(len(h_cross)):
		seg = line_segmentation(h_cross[i], th_z, th_x)
		h_seg_temp = []
		for j in range(len(seg)):
			h_seg_temp.append(seg[j])
		h_seg.append(h_seg_temp)
	h_seg_cross = []
	for i in range(len(h_seg)):
		seg_temp = []
		for j in range(len(h_seg[i])):
			seg_temp += h_seg[i][j]
		h_seg_cross.append(seg_temp)

	cross_line = findCross(h_seg_cross, f_cross)
	top_line = []
	bottom_line = []
	error_line_flag = False

	# 对交线进行上下盘提取
	if(faultName == 'f10_2017'):
		for i in range(len(cross_line)):
			bottom_line.append(cross_line[i])
			top_line.append(cross_line[i])
	else:
		for i in range(int(len(cross_line) / 2)):
			if(cross_line[i * 2][2] < cross_line[i * 2 + 1][2]):
				bottom_line.append(cross_line[i * 2])
				top_line.append(cross_line[i * 2 + 1])
			else:
				top_line.append(cross_line[i * 2])
				bottom_line.append(cross_line[i * 2  + 1])
			if(top_line[i][0] == bottom_line[i][0]):
				continue
			else:
				error_line_flag = True
		if(int(len(cross_line) / 2) == 0):
			printError(str(listNum) + ':' + layerName + '_' + faultName + '无交线')
			return 
	
	# 层位数据预处理
	afterLayer = layerPreprocess(h_seg_cross, f_cross, top_line, bottom_line)
	# 保存交线图片
	plt.figure(faultName+str(i))
	y = np.array(bottom_line).T[0]
	z = np.array(bottom_line).T[1]
	y2 = np.array(top_line).T[0]
	z2 = np.array(top_line).T[1]
	plt.scatter(y, z, s=1, c='r')
	plt.scatter(y2,z2, s=1, c='g')
	path = './data/cross/'+ layerName + '/' + faultName
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
	plt.savefig(path + '/'+  'line' + '.png')
	plt.savefig('./data/cross/lines/' + layerName +'+'+ faultName  + '.png')


	# 检测错误交线
	if(len(top_line) != len(bottom_line)):
		error_line_flag = True
	if(error_line_flag):
		plt.savefig('./data/cross/lines_error/' + layerName +'+'+ faultName  + '.png')

	np.savetxt('data/交线/' + layerName + '-' + faultName  + '-L1.txt', top_line)
	np.savetxt('data/交线/' + layerName + '-' + faultName  + '-L2.txt', bottom_line)

	# 保存预处理后的层位数据
	out_layer = []
	out_fault = []
	for i in range(len(afterLayer)):
		out_layer = out_layer + afterLayer[i]
	for i in range(len(f_cross)):
		out_fault = out_fault + f_cross[i]
	np.savetxt('data/预处理/层位/' + layerName + '.txt', out_layer)
	np.savetxt('data/预处理/断层/' + faultName  + '.txt', out_fault)



	# 一个断层面与层位相交的所有剖面显示
	for i in range(len(f_cross)):
		if(f_cross[i]):
			plt.figure(str(i) + faultName+':'+str(np.array(f_cross[i]).T[0][0]))
			x = np.array(f_cross[i]).T[0]
			y = np.array(f_cross[i]).T[1]
			z = np.array(f_cross[i]).T[2]
			y2 = np.array(h_cross[i]).T[1]
			z2 = np.array(h_cross[i]).T[2]
			y3 = np.array(afterLayer[i]).T[1]
			z3 = np.array(afterLayer[i]).T[2]
			plt.scatter(y, z, s=1, c='r')
			# plt.scatter(y2, z2, s=1, c='g')
			plt.scatter(y3, z3, s=1, c='g')
			for o in range(len(cross_line)):
				if(cross_line[o][0] == x[0]):
					plt.scatter(cross_line[o][1],cross_line[o][2], s=3, c='b')

			plt.savefig(path + '/'+ str(i) +  '+' +str(np.array(f_cross[i]).T[0][0]) + '.png')
			plt.savefig('./data/cross/all/'+ layerName + '+' + faultName + '+'+  str(np.array(f_cross[i]).T[0][0]) + '.png')
			plt.close() 
	printSuccess(str(listNum) + ':' +layerName + '-' + faultName + '交线提取完成！')
	
	return 

if __name__ == '__main__':
	catalog = Catalog()
	for q in range(5, 6):
		layer = catalog.layers[q]
		fault = catalog.faults
		for i in range(0, len(fault)):
			Intersction(layer, fault[i], i)
			plt.close('all') 

