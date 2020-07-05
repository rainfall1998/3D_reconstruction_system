#使用graduation环境，用来处理colmap dense重建得到的数据
import cv2
import numpy as np 
#将图片resize并转化为npy文件
def get_data(img_path, write_path):
    img = cv2.imread(img_path)
    
    img_resize = cv2.resize(img,(640,480),cv2.INTER_NEAREST)
    #备注：resize的方法
    # INTER_NEAREST   最近邻插值
    # INTER_LINEAR    双线性插值（默认设置）
    # INTER_AREA  使用像素区域关系进行重采样。
    # INTER_CUBIC 4x4像素邻域的双三次插值
    # INTER_LANCZOS4  8x8像素邻域的Lanczos插值
    depth = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    np.save(write_path, np.array(depth))

###############################################main 区域
depth_path = './depth/8_nprmal.png'
npy_path = './npy/8_normal.npy'
get_data(depth_path,npy_path)
