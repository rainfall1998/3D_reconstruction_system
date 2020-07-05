file = open('val_files.txt','w')
num_max = 200
for i in range(0,num_max):
	path = '/home/rainbow/niu/data/interiornet_monodepth/camera_1_1/rgb/ '
	line = path + str(i) + '\n'
	#line_2 = path + str(999-i) + '\n'
	file.write(line)
	#file.write(line_2)