#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import numpy as np
import scipy.misc
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import time
import math
#############################################
#文件说明：
#depth 采用.png格式的图像输入
#使用需要修改三个个地址：点保存地址，ply文件保存地址，深度输入地址以及彩色图像地址，在main里面修改
#注意修改相机参数
##############################################

#################################################
#输入深度，得到三维空间坐标，filename为储存坐标的名字
#  需要修改fx,fy,cx,cy
#################################################
def gentxtcord(filename,depth):
    fx = 600.391
    fy = 600.079
    cx = 320
    cy = 240
    xcord = []
    ycord = []
    zcord = []
    j = 0
    with open(filename,'w') as f:
        for j in range(0,480):
            for i in range(0,640):
                Z = depth[j,i]
                X = (i - cx)/fx*Z
                Y = (j - cy)/fy*Z
                xcord.append(X)
                ycord.append(Y)
                zcord.append(Z) 
                line = str(X) + ',' + str(Y) + ',' + str(Z) + '\n'
                f.write(line)
    return [xcord,ycord,zcord]
	
#################################################################################可视化部分

##############################################
#彩色点云PLY生成
#
#输入三维点坐标序列gtxyz
#彩色图片地址imgpath
#写入ply地址pc_file
##############################################
def genply_noRGB(gtxyz,imgpath,pc_file):
    # add imgRGB
    t1=time.time()
    imgRGB = Image.open(imgpath)
    width,height = imgRGB.size[0], imgRGB.size[1]
    df=np.zeros((6,width*height))
    df[0] = gtxyz[0]
    df[1] = gtxyz[1]
    df[2] = gtxyz[2]
    img = np.array(imgRGB)
    df[3] = img[:, :, 0:1].reshape(-1)
    df[4] = img[:, :, 1:2].reshape(-1)
    df[5] = img[:, :, 2:3].reshape(-1)
    float_formatter = lambda x: "%.4f" % x
    points =[]
    for i in df.T:
        points.append("{} {} {} {} {} {} 0\n".format
                      (float_formatter(i[0]), float_formatter(i[1]), float_formatter(i[2]),
                       int(i[3]), int(i[4]), int(i[5])))
    file = open(pc_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()

    t2=time.time()
    print("Write into .ply file Done.",t2-t1)
##############################################
#无颜色点云PLY生成
#
#输入三维点坐标序列gtxyz
#写入ply地址pc_file
##############################################
def genply_RGB(gtxyz,pc_file):
    lenth_point = len(gtxyz[0])
    df=np.zeros((3,lenth_point))
    df[0] = gtxyz[0]
    df[1] = gtxyz[1]
    df[2] = gtxyz[2]
    # df[3] = np.zeros(lenth_point)
    # df[4] = np.zeros(lenth_point)
    # df[5] = np.zeros(lenth_point)
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
##########################################################################主流程
def main():
    # load gt and pred depth map
    num = 24
    depth_path = './depth/'+str(num)+'.png'
    point_path = './point/'+str(num)+'.txt'
    imgpath = './img/'+str(num)+'.png'
    pc_file = './ply/'+str(num)+'.ply'
    gt = cv.imread(depth_path,cv.IMREAD_UNCHANGED)
    gray_img = gt[:,:,1]
    gt_cord = gentxtcord(point_path,gray_img)
    genply_RGB(gt_cord,imgpath,pc_file)

if __name__ == '__main__':
    main()
