import numpy as np 
import time
from scipy.spatial.transform import Rotation as R
###########################
# little tool
##############################
def str_tofloat(data):
	transfer = map(np.float,data)
	return np.array(list(transfer))
def point_camera(p1,r_inverse):
	p_world = np.dot(r_inverse ,(p1).T)#  - t.T
	return np.array(p_world.T)
def get_r(q):
	r = np.zeros((3,3))
	r[0,0] = 1-2*q[2]*q[2]-2*q[3]*q[3]
	r[1,1] = 1-2*q[1]*q[1]-2*q[3]*q[3]
	r[2,2] = 1-2*q[1]*q[1]-2*q[2]*q[2]
	r[0,1] = 2*q[1]*q[2]-2*q[0]*q[3]
	r[0,2] = 2*q[1]*q[3]+2*q[0]*q[2]
	r[1,0] = 2*q[1]*q[2]+2*q[0]*q[3]
	r[1,2] = 2*q[2]*q[3]-2*q[0]*q[1]
	r[2,0] = 2*q[1]*q[3]-2*q[0]*q[2]
	r[2,1] = 2*q[2]*q[3]+2*q[0]*q[1]
	r_inverse = np.matrix(r).I
	return r_inverse
def scipy_transfer(quat):
	r = R.from_quat(quat)
	return np.matrix(r.as_matrix()).I
############################
#visual and getT
############################
#T input
def get_T(path_txt):
	T = np.zeros((4,4))
	file_T = open(path_txt,'r')
	for i in range(0,4):
		line = file_T.readline()
		data_T = str_tofloat(line.split())
		T[i,0] = data_T[0]
		T[i,1] = data_T[1]
		T[i,2] = data_T[2]
		T[i,3] = data_T[3]
	return T

#visual
def genply(gtxyz,pc_file,lenth_point):
    df=np.zeros((3,lenth_point))
    df[0] = gtxyz[0]
    df[1] = gtxyz[1]
    df[2] = gtxyz[2]
    float_formatter = lambda x: "%.4f" % x
    points =[]
    for i in df.T:
        points.append("{} {} {} \n".format
                      (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]) ))
    file = open(pc_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()

    print("Write into .ply file Done.")

#输入point记录的txt,输出xyzrecord和world_point
def local_world(path_local,file_write,T,xcord,ycord,zcord,flag):
	print('start transfer')
	file_local = open(path_local,'r')
	while True:
		line_p = file_local.readline()
		if not line_p:
			break
			pass
		data_p = line_p.split(',')
		point_ca = np.ones(4)
		point_ca[0:3] = str_tofloat(data_p[0:3])
		if flag:
			point_world = point_camera(point_ca,T)
			xcord.append(point_world[0])
			ycord.append(point_world[1])
			zcord.append(point_world[2])
			line_point_world = str(point_world[0]) + ',' + str(point_world[1]) + ',' + str(point_world[2]) + '\n'
		else:
			point_world = point_ca
			xcord.append(point_world[0])
			ycord.append(point_world[1])
			zcord.append(point_world[2])
			line_point_world = str(point_world[0]) + ',' + str(point_world[1]) + ',' + str(point_world[2]) + '\n'
		# print(point_world[1,0])
		# print(type(point_world[0]))
		# break
		file_write.write(line_point_world)

path_T = 'T_data.txt'
path_world = './point_world/03_testT.txt'
path_ply = './ply/icp/024.ply'
T = get_T(path_T)
xcord = []
ycord = []
zcord = []
file_w = open(path_world,'w')
local_world('./point/0.txt',file_w,T,xcord,ycord,zcord,False)
local_world('./point/24.txt',file_w,T,xcord,ycord,zcord,True)
length = len(xcord)
genply([xcord,ycord,zcord],path_ply,length)
