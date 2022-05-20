import numpy as np
import Utils as utils
def load_np(name):
  data = np.loadtxt(name)
  res = []
  for i in range(len(data)):
    if(data[i][2] < 0):
      data[i][2] = -data[i][2]
    res.append([
      data[i][1],
      data[i][0],
      data[i][2]
    ])
  return res
def load_np_corss(name):
  data = np.loadtxt(name)
  res = []
  for i in range(len(data)):
    if(data[i][2] < 0):
      data[i][2] = -data[i][2]
    res.append([
      data[i][1],
      data[i][0],
      data[i][2]
    ])
  return res
	