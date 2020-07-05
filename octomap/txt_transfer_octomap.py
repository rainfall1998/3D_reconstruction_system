import numpy as np 
import octomap

######################################
#   little tools
######################################
#格式转换
def str_tofloat(data):
	transfer = map(np.float,data)
	return np.array(list(transfer))

######################################
#    主要流程
######################################

def txt_read(file_path,tree):
    generation = 0
    file = open(file_path,'r')
    while True:
        line = file.readline()
        if not line:
            break
            pass
        data = str_tofloat(line.split(','))
        tree.updateNode(data,True)
        if generation%100000 == 0:
            print('the generation: ',generation)
        generation += 1
    #return point.T

file_txt = '/media/rainfall/TOSHIBA EXT/graduation/data/point/code/pfm2txt/point_world/5_22_31_changeyz_worldpoint.txt'
file_bt = './bt/airsim_5_22_31_changeyz_worldpoint.bt'
tree = octomap.OcTree(0.1)
txt_read(file_txt,tree)
tree.updateInnerOccupancy()
tree.writeBinary(bytes(file_bt,encoding='utf-8'))