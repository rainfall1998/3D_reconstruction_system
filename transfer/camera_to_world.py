#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import scipy.misc
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2 as cv
import time
from scipy.spatial.transform import Rotation as R
import math
#############################################
#文件说明：
#depth 采用.png格式的图像输入
#使用需要修改三个个地址：点保存地址，ply文件保存地址，深度输入地址
#注意修改相机参数
##############################################




######################################
#   little tools
######################################
#格式转换
def str_tofloat(data):
	transfer = map(np.float,data)
	return np.array(list(transfer))
#sfm to npy
def sfm2npy(transfer_name):	
	pfm_path = './pfm/' + transfer_name + '.pfm'
	npy_path = './npy/' + transfer_name + '.npy'
	mat = cv.imread(pfm_path)
	print(type(mat))
	np.save(npy_path,mat)
	return npy_path
#get r 
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
#local to the world
def point_camera(p1,r_inverse,t):
	p_world = np.dot(r_inverse ,(p1-t).T)#  - t.T
	return np.array(p_world.T)


#######################################
# 转化函数
#######################################

#输入深度，得到相机坐标系的三维空间坐标，filename为储存坐标的名字
def gentxtcord(filename,depth):
	fx = 600.391
	fy = 600.079
	cx = 320
	cy = 240
	j = 0
	with open(filename,'w') as f:
		for col in depth:
			i = 0
			for Z in col:
				Z =Z
				X = (i - cx)/fx*Z
				Y = (j - cy)/fy*Z
				line = str(X) + ',' + str(Y) + ',' + str(Z) + '\n'
				f.write(line)
				i += 1
			j += 1

#得到世界坐标系
def get_pointdata(p_path,q,t,  xcord,ycord,zcord):
	point_world_path = './point_world/small_worldpoint_5_23_5.txt'
	file_p = open(p_path,'r')
	file_write = open(point_world_path,'w')
	point_ca = np.zeros(3)
	r = scipy_transfer(q)
	while True:
		line_p = file_p.readline()
		if not line_p:
			break
			pass
		data_p = line_p.split(',')
		point_ca = str_tofloat(data_p[0:3])
		point_world = point_camera(point_ca,r,t)
		xcord.append(point_world[0,0])
		ycord.append(point_world[1,0])
		zcord.append(point_world[2,0])
		line_point_world = str(point_world[0,0]) + ',' + str(point_world[1,0]) + ',' + str(point_world[2,0]) + '\n'
		file_write.write(line_point_world)
	file_write.close()


######################################
# 可视化
######################################
#ply文件写入，需要转化后的三维坐标和图像地址
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
######################################
# 流程函数
######################################
def get_file_name(qt_path):
	file_q = open(qt_path,'r')
	line_q = file_q.readline()
	data_q = []
	q = np.zeros(4)
	t = np.zeros(3)
	xcord = []
	ycord = []
	zcord = []
	print('data start transfer')
	generation = 0
	while True:
		t1=time.time()
		line_q = file_q.readline()
		if not line_q:
			break
			pass
		data_q = line_q.split(',')
		t = str_tofloat(data_q[1:4])
		q = str_tofloat(data_q[4:8])
		path_png = data_q[8]
		# read depth
		depth =cv.imread('./depth/'+path_png,cv.IMREAD_GRAYSCALE)
		#print(depth.shape)
		# transfer to the local point
		local_point_path = './point/'+ path_png[0:-4] + '.txt'
		gt_cord = gentxtcord(local_point_path,depth)
		# tranfer the local to the world 
		get_pointdata(local_point_path,q,t,xcord,ycord,zcord)
		generation+=1
		if generation % 1 == 0:
			t2=time.time()
			print('##################')
			print("two epoch cost .",t2-t1)
			print('the picture generation is: ',generation)
	lenth_point = len(xcord)
	genply([xcord,ycord,zcord],'./ply/small_035_p8.ply',lenth_point)			


##########################################################################主流程
def main():
    qt_path = './camera_pose/image_colmap_simi_2.txt'
    get_file_name(qt_path)

if __name__ == '__main__':
    main()