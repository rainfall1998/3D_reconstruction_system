#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK
#include "rpc/rpc_error.h"
STRICT_MODE_ON

#include <Windows.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include "common/common_utils/FileSystem.hpp"
#include "controllers/PidController.hpp"
#include "sensors/gps/GpsBase.hpp"
#include "common/EarthUtils.hpp"
//#include "testfunc.h"              // add the test function of our team

#include "image_construction.h"
#include "AffineTransform.h"
#include "detect_num_nav.h"
#include "kalman_filter.h"
#include "detect_circle.h"
#include "ImageToMat.h"
#include "detect_num.h"
#include "Ellipse.h"
#include "detect_tree.h"
#include "CollectDataVO.h"
#define WINDOW_NAME "[效果图窗口]"        //为窗口标题定义的宏 

//相机内参
const double camera_cx = 319.5;
const double camera_cy = 239.5; 
const double camera_fx = 269.5; 
const double camera_fy = 269.5;
using namespace cv;
using namespace std;
//像素点到相机中心的实际误差
Vec2i projectPixelToReal(float alt, Vec2i& target)
{
	float x, y;
	Vec2i real_xy;
	x = alt*(target[0] - camera_cx) / camera_fx;
	y = alt*(target[1] - camera_cy) / camera_fy;
	real_xy[0] = x;
	real_xy[1] = y;
	return real_xy;
}
Mat convertTo3Channels(const Mat& binImg)
{
	Mat three_channel = Mat::zeros(binImg.rows, binImg.cols, CV_32FC3);
	std::vector<Mat> channels;
	for (int i = 0; i < 3; i++)
	{
		channels.push_back(binImg);
	}
	merge(channels, three_channel);
	return three_channel;
}

Ptr<cv::ml::ANN_MLP> bp = cv::ml::ANN_MLP::create();

const float pi = 3.1415926535897;
float Q = 0.2;   //the process noise variance, used as a parameter for tuning the intensity of the filter action
float Q1 = 0.2;
const float R = 0.2;  //the sensor noise variance
const float R1 = 0.0000047;  //the sensor noise variance--经度
const float R2 = 0.000004;  //the sensor noise variance--纬度
#define CLIP3(_n1, _n,  _n2) {if (_n<_n1) _n=_n1;  if (_n>_n2) _n=_n2;}
void SaveResult(cv::Mat & img, ImageCaptureBase::ImageResponse & response, std::vector< Point2f > corner, int ArucoID)// ????? this fuction haven't used
{
	char savepathpng[100], savepathpmf[100], savepathtxt[100];
	ofstream resultID;
	cv::Mat TempImg;
	cv::Mat imgCodecopy = img.clone();			//保存原图片
	int width = corner.at(1).x - corner.at(0).x;
	int height = corner.at(3).y - corner.at(0).y;
	cv::Mat imageROI = imgCodecopy(Rect(corner.at(0).x, corner.at(0).y, width, height));	
	imageROI.convertTo(TempImg, TempImg.type());		
	sprintf(savepathpng, "D:\\images\\%d.png", ArucoID);
	sprintf(savepathpmf, "D:\\depth\\%d.pfm", ArucoID);
	cv::imwrite(savepathpng, imgCodecopy);
	resultID.open("D:\\result.txt",ios::app);
	resultID << ArucoID << " " << corner.at(0).x << " " << corner.at(0).y
		<< " " << corner.at(2).x << " " << corner.at(2).y << std::endl;
	resultID.close();
	Utils::writePfmFile(response.image_data_float.data(), response.width, response.height,
		savepathpmf);
}
//test the circle
void TestRedAreaDetect() {
	//char filename[100];
	std::string filename = "D:\\ROS\\Competition0730_me\\picture\\aresult_metest.jpg";
	cv::Mat whole_pic = imread(filename);
	cv::imshow("whole", whole_pic);//error cannot read
	waitKey(0);
	cv::Mat bin_img;
	//Circle_ImageThreshold(whole_pic, bin_img);
	cv::imshow("out", bin_img);
	waitKey(0);
}
//////////////////count the local position
// output the picture and add the param in the picture（not test）
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{

	Mat& image = *(cv::Mat*) param;
	switch (event)
	{
		//左键按下消息
	case EVENT_LBUTTONDOWN:
	{

		cout << static_cast<int>(image.at<Vec3b>(y, x)[0]) << ",";
		cout << static_cast<int>(image.at<Vec3b>(y, x)[1]) << ",";
		cout << static_cast<int>(image.at<Vec3b>(y, x)[2]) << endl;
		cout << "点的坐标为y: "<< y << endl;
		cout << "点的坐标为x: " << x << endl;
	}
	break;
	}
}

////////////////////////控制函数公用
void fstay(msr::airlib::MultirotorRpcLibClient &client, float time = 0.01)//保持飞机高度悬停策略
{
	client.moveByAngleThrottle(0.0f, 0.0f, (float)0.587, 0.0f, (float)time);
	std::this_thread::sleep_for(std::chrono::duration<double>(time));
	client.hover();
}
void xstay(msr::airlib::MultirotorRpcLibClient &client, float time = 0.01)//保持飞机悬停策略
{
	client.hover();
	std::this_thread::sleep_for(std::chrono::duration<double>(time));
}
int xcontrol(msr::airlib::MultirotorRpcLibClient &client, int num, float pitch_d, float roll_d, float time = 0.14)
{
	int count = 0;
	while (1) {
		client.moveByAngleThrottle(-pitch_d, -roll_d, 0.588, 0.0f, time);
		std::this_thread::sleep_for(std::chrono::duration<double>(time));
		client.hover();//悬停
		xstay(client, 0.5);
		count++;
		if (count >= num)
		{
			xstay(client, 1.0);
			return 1;
		}
	}

}
//深度图最小深度获得
struct get_img
{
	ImageToMat *img2mat;

	void start(ImageToMat *img2mat_in)
	{
		img2mat = img2mat_in;
	}
	float get_distance()
	{
		//data type
		typedef ImageCaptureBase::ImageResponse ImageResponse;
       //读图
		std::vector<ImageResponse> response = (*img2mat).get_depth_mat();

		float Max_distance = 0;
	
		for (int iter_num = 1; iter_num < 640*480-1; iter_num++)
		{

			if (response.at(0).image_data_float.at(iter_num) == 255) continue;
			if (Max_distance > response.at(0).image_data_float.at(iter_num) &&
				response.at(0).image_data_float.at(iter_num) <= 5)
				Max_distance = response.at(0).image_data_float.at(iter_num);

		}
		return Max_distance;
	}
	void get_front(cv::Mat3b &image_front)
	{
		image_front = (*img2mat).get_front_mat();
	}
	void get_depth(std::vector<ImageCaptureBase::ImageResponse> &response)
	{
		response = (*img2mat).get_depth_mat();
	}
	


};
// 高度获得函数，使用：get_height(prev,client,Barometer_origin)
struct get_data
{
	msr::airlib::MultirotorRpcLibClient *client;
	BarometerData Barometer_origin;
	prev_states* prev;
	void start(prev_states* prev_in, msr::airlib::MultirotorRpcLibClient *client_in, BarometerData Barometer_in)
	{
		prev = prev_in;
		client = client_in;
		Barometer_origin = Barometer_in;
	}
    float get_height()
	{
		BarometerData curr_bardata = (*client).getBarometerdata();//气压计数据
		float cur_height = curr_bardata.altitude - Barometer_origin.altitude;//世界坐标系
		cur_height = kalman(cur_height, prev, 0, Q, R);//滤波 the height
		return cur_height;
	}
};
//运动数据结构体
struct data_circle {
	float _x, _y, h;
	float pitch, roll;
	float time_p, time_r;

	int solve(float dist)
	{
		// 注意到abc中除c均恒正，对称轴必定为负，所以取大根就可以了。不过这时y一定得大于0.0039（显然）
		float solution = 0;
		float delta = 0;
		float a = 0.454;
		float b = 0.8191;
		float c = 0.0039 - dist;
		delta = sqrt(b*b - 4 * a*c);
		solution = (delta - b) / 2 / a;
		return(solution);
	}

	bool start(float x, float y, float height = 0)
	{
		_x = x;
		_y = y;
		h = height;
		pitch = 0.1*_x / abs(_x);
		roll = 0.1*_y / abs(_y);
		time_p = solve(abs(_x));
		time_r = solve(abs(_y));
		return true;
	}
};
//用于单独运动的使用
data_circle sig_move;


////////////////////////二维码补充函数
//拍照高度控制函数
int pillar_height_origin(float num, prev_states* prev, msr::airlib::MultirotorRpcLibClient &client, BarometerData &Barometer_origin, PidController &pidZ)
{
	float gravity_d = 0.6;// 平衡量
	int stay = 0;// 保证稳定停止量
	int ori_h = 1, now_h = 1;
	bool h_switch = false;

	BarometerData curr_bardata = client.getBarometerdata();//气压计数据
	float cur_height = curr_bardata.altitude - Barometer_origin.altitude;//世界坐标系
	cur_height = kalman(cur_height, prev, 0, Q, R);//滤波 the height
	if (num > cur_height)
		ori_h = 1;
	else
		ori_h = 0;

	while (1) {
		cout << "\n the test control mode\n";
		////the Begin action
		curr_bardata = client.getBarometerdata();//气压计数据
		cur_height = curr_bardata.altitude - Barometer_origin.altitude;//世界坐标系
		cur_height = kalman(cur_height, prev, 0, Q, R);//滤波 the height
		std::cout << "ned_curr: " << cur_height << std::endl;
		float delta_throttle = pidZ.control(cur_height) + 0.6;//control the A
		CLIP3(0.4, delta_throttle, 0.8);//取中间值
		client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);// THE mid clent
		std::this_thread::sleep_for(std::chrono::milliseconds(10));// stop the other action
																   // the height fly
		pidZ.setPoint(num, 0.3, 0, 0.4);
		// judge the flow direction
		if (num > cur_height)
		{
			gravity_d = 0.587;
			now_h = 1;
		}
		else
		{
			gravity_d = 0.58;
			now_h = 0;
		}
		delta_throttle = pidZ.control(cur_height) + gravity_d;
		CLIP3(0.4, delta_throttle, 0.8);//取中间值
		client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.2f);// 
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		// stop the action
		if (ori_h == now_h)
			h_switch = false;
		else
			h_switch = true;
		if ((num - cur_height) < 0.1 && (num - cur_height) > -0.1 || h_switch)
		{
			std::cout << "\naim is ready\n";
			client.hover();
			xstay(client, 0.5);
			client.hover();
			return 1;
		}
	}

}
int pillar_height(float num, prev_states* prev, msr::airlib::MultirotorRpcLibClient &client, BarometerData &Barometer_origin, PidController &pidZ)
{
	float gravity_d = 0.6;// 平衡量
	int stay = 0;// 保证稳定停止量
	int ori_h = 1, now_h = 1;
	bool h_switch = false;

	BarometerData curr_bardata = client.getBarometerdata();//气压计数据
	float cur_height = curr_bardata.altitude - Barometer_origin.altitude;//世界坐标系
	cur_height = kalman(cur_height, prev, 0, Q, R);//滤波 the height
	if (num > cur_height)
		ori_h = 1;
	else
		ori_h = 0;

	while (1) {
		std::cout << "\n the test control mode\n";
		////the Begin action
		curr_bardata = client.getBarometerdata();//气压计数据
		cur_height = curr_bardata.altitude - Barometer_origin.altitude;//世界坐标系
		cur_height = kalman(cur_height, prev, 0, Q, R);//滤波 the height
		std::cout << "ned_curr: " << cur_height << std::endl;
		float delta_throttle = pidZ.control(cur_height) + 0.6;//control the A
		CLIP3(0.4, delta_throttle, 0.8);//取中间值
		client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);// THE mid clent
		std::this_thread::sleep_for(std::chrono::milliseconds(10));// stop the other action
																   // the height fly
		pidZ.setPoint(num, 0.6, 0, 0.4);
		// judge the flow direction
		if (num > cur_height)
		{
			if (num > cur_height + 3)
				gravity_d = 0.7;
			else
				gravity_d = 0.587;
			now_h = 1;
		}
		else
		{
			if (num > cur_height - 3)
				gravity_d = 0.58;
			else
				gravity_d = 0.45;
			now_h = 0;
		}
		delta_throttle = pidZ.control(cur_height) + gravity_d;
		CLIP3(0.4, delta_throttle, 0.8);//取中间值
		client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.2f);// 
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		// stop the action
		if (ori_h == now_h)
			h_switch = false;
		else
			h_switch = true;
		if ((num - cur_height) < 0.1 && (num - cur_height) > -0.1 || h_switch)
		{
			std::cout << "\naim is ready\n";
			client.hover();
			xstay(client, 0.5);
			client.hover();
			return 1;
		}
	}

}
//斜飞控制函数 //用在一个地方，一个是多边形检测函数里挪到中心
void slopefly(msr::airlib::MultirotorRpcLibClient &client, int num, float pitch_d, float roll_d, float time = 0.14)
{
	float pitch, roll;
	if (abs(pitch_d) < 3)
	{
		pitch = 0;
	}
	else
	{
		pitch = 0.1 * pitch_d / abs(pitch_d);
	}
	if (abs(roll_d) < 3)
	{
		roll = 0;
	}
	else
	{
		roll = 0.1 * roll_d / abs(roll_d);
	}
	client.moveByAngleThrottle(pitch, roll, 0.588, 0.0f, time);
	std::this_thread::sleep_for(std::chrono::duration<double>(time));
	client.hover();//悬停
	xstay(client, 0.4);
}
//斜飞控制函数 //用在首位停机坪识别
void slopefly_PARK(msr::airlib::MultirotorRpcLibClient &client, int num, float pitch_d, float roll_d, float time = 0.14)
{
	float pitch, roll;
	int flag_move = 0;//用于判断移动时间和幅度
	if (abs(pitch_d) < 3)
	{
		pitch = 0;
	}
	else if(abs(pitch_d) < 20 && abs(pitch_d) >= 3)
	{
		pitch = 0.1 * pitch_d / abs(pitch_d);
	}
	else if (abs(pitch_d) >= 20)
	{
		pitch = 0.2 * pitch_d / abs(pitch_d);
		flag_move = 1;
	}
	//roll
	if (abs(roll_d) < 3)
	{
		roll = 0;
	}
	else if (abs(roll_d) < 20 && abs(roll_d) >= 3)
	{
		roll = 0.1 * roll_d / abs(roll_d);
	}
	else if (abs(roll_d) >= 20)
	{
		roll = 0.2 * roll_d / abs(roll_d);
		flag_move = 1;
	}
	if(flag_move == 0)
		client.moveByAngleThrottle(pitch, roll, 0.588, 0.0f, time);
	else
		client.moveByAngleThrottle(pitch, roll, 0.590, 0.0f, time);
	std::this_thread::sleep_for(std::chrono::duration<double>(time));
	client.hover();//悬停
	xstay(client, 0.4);
}
//多边形检测函数，面积优先，注意识别数量《40
void detect_squares_center(Mat &src, std::vector<Vec2i> &tree_circles)
{
	Mat src_gray;
	int thresh = 195;
	int max_thresh = 255;
	RNG rng(12345);

	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	Mat src_copy = src.clone();
	Mat threshold_output;
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	// 对图像进行二值化
	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
	// 寻找轮廓
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//对每个轮廓计算其凸包
	std::vector<std::vector<Point> >poly(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), poly[i], 5, true);
	}
	//////显示识别到的轮廓
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, std::vector<Vec4i>(), 0, Point());
	}
	//namedWindow("Hull demo", CV_WINDOW_AUTOSIZE);
	//imshow("Hull demo", drawing);

	//计算轮廓面积，并进行排序
	float area;
	float max_area = 0;
	int area_index = 0;
	Point area_data;//x是面积，y是index
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		/*if ((area > max_area)&&(poly[i].size()<35))*/
			if ((area > max_area) && (poly[i].size()<35))
		{
			area_index = i;
			max_area = area;
		}
		
	}

	// 选取最大的那个面积坐标（或者进行排序）
			double xx = 0; double yy = 0;
			if (contours.size() > 0)
			{
				if (poly[area_index].size() > 2)
				{
					for (int j = 0; j < poly[area_index].size(); j++)
					{
						xx = xx + poly[area_index][j].x;
						yy = yy + poly[area_index][j].y;
					}
				}
				Vec2i XY;

				XY[0] = xx / poly[area_index].size();
				XY[1] = yy / poly[area_index].size();
				if (XY[0] != 0 && XY[1] != 0)
				{
					tree_circles.push_back(XY);
				}
			}
}
//停机坪检测函数
void detect_squares_PARK(Mat &src, std::vector<Vec2i> &tree_circles)
{
	Mat src_gray;
	int thresh = 195;
	int max_thresh = 255;
	RNG rng(12345);

	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));
	Mat src_copy = src.clone();
	Mat threshold_output;
	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	// 对图像进行二值化
	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
	// 寻找轮廓
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//对每个轮廓计算其凸包
	std::vector<std::vector<Point> >poly(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), poly[i], 5, true);
	}
	//////显示识别到的轮廓
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, std::vector<Vec4i>(), 0, Point());
	}
	/*namedWindow("Hull demo", CV_WINDOW_AUTOSIZE);
	imshow("Hull demo", drawing);*/

	//计算轮廓面积，并进行排序
	float area;
	float max_area = 0;
	int area_index = 0;
	Point area_data;//x是面积，y是index
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		if ((area > max_area) && (poly[i].size()<15))
		{
			area_index = i;
			max_area = area;
		}

	}

	// 选取最大的那个面积坐标（或者进行排序）
	double xx = 0; double yy = 0;
	if (contours.size() > 0)
	{
		if (poly[area_index].size() > 2)
		{
			for (int j = 0; j < poly[area_index].size(); j++)
			{
				xx = xx + poly[area_index][j].x;
				yy = yy + poly[area_index][j].y;
			}
		}
		Vec2i XY;

		XY[0] = xx / poly[area_index].size();
		XY[1] = yy / poly[area_index].size();
		if (XY[0] != 0 && XY[1] != 0)
		{
			tree_circles.push_back(XY);
		}
	}
}
//停机坪识别函数
bool platform_move_PARK(ImageToMat &img2mat, int &count_stop, PidController &pidP_Y, PidController &pidP_X, msr::airlib::MultirotorRpcLibClient &client,int search_direction = -2)
{
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	Vec4i _UpRight;//x,y,a,b
	cv::Mat3b image_bellow = img2mat.get_below_mat();//得到底部图
	std::vector<Vec2i> plat_location;// output
	detect_squares_PARK(image_bellow, plat_location);

	if (plat_location.size() > 0)
	{
		float py_location = plat_location[0][0];
		float px_location = plat_location[0][1];
		cout << "\nthe pillar is in the scene\n";
		Point XY;
		XY.y = px_location;
		XY.x = py_location;
		cv::circle(image_bellow, XY, 4, cv::Scalar(0, 0, 255));
		/*cv::namedWindow("imageScene");
		cv::imshow("imageScene", image_bellow);
		cv::waitKey(1);*/
		//save the picture
		std::cout << "x_pixel: " << px_location << "  y_pixel: " << py_location << std::endl;
		/////如果中心吻合，退出
		if (abs(px_location - 240) < 10 && abs(py_location - 320) < 10)
		{
			count_stop++;
			if (count_stop > 1)
			{
				std::cout << "\n the pillar is recognized\n";
				return true;
			}
		}
		else
		{
			////////// 调整使无人机正对停机坪位置
			std::cout << "pixel...." << std::endl;
			float roll = py_location - 320;
			float pitch = px_location - 240;
			std::cout << "x_pixel: " << pitch << "  y_pixel: " << roll << std::endl;
			slopefly_PARK(client, 1, pitch, roll);
			client.hover();
			return false;
		}
	}
	else
	{
		std::cout << "\n the pillar wasn't recognized\n";
		//搜不到停机坪，向后移动
		sig_move.start(search_direction, 0);
		xcontrol(client, 1, sig_move.pitch, 0.0, sig_move.time_p);//x
		client.hover();
		xstay(client, 1.0);
		client.hover();
		return false;
	}


}
// pillar识别并移动到固定位置
bool platform_move(ImageToMat &img2mat, int &count_stop, PidController &pidP_Y, PidController &pidP_X, msr::airlib::MultirotorRpcLibClient &client)
{
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	Vec4i _UpRight;//x,y,a,b
	cv::Mat3b image_bellow = img2mat.get_below_mat();//得到底部图
	std::vector<Vec2i> plat_location;// output
	detect_squares_center(image_bellow, plat_location);

	if (plat_location.size() > 0)
	{
		float py_location = plat_location[0][0];
		float px_location = plat_location[0][1];
		cout << "\nthe pillar is in the scene\n";
		Point XY;
		XY.y = px_location;
		XY.x = py_location;
		cv::circle(image_bellow, XY, 4, cv::Scalar(0, 0, 255));
	/*	cv::namedWindow("imageScene");
		cv::imshow("imageScene", image_bellow);
		cv::waitKey(1);*/
		//save the picture
		/////如果中心吻合，退出
		if (abs(px_location - 240) < 5 && abs(py_location - 280) < 10)//x方向修改成了5
		{
			count_stop++;
			if (count_stop > 2)
			{
				std::cout << "\n the pillar is recognized\n";
				return true;
			}
		}
		else
		{
			////////// 调整使无人机正对停机坪位置
			std::cout << "pixel...." << std::endl;
			float roll = py_location - 280;
			float pitch = px_location - 240;
			std::cout << "x_pixel: " << pitch << "  y_pixel: " << roll << std::endl;
			slopefly_PARK(client, 1, pitch, roll);
			client.hover();
			return false;
		}
	}
	else
	{
		std::cout << "\n the pillar wasn't recognized\n";
		return true;
	}
}

bool platform_moveRough(ImageToMat &img2mat, int &count_stop, PidController &pidP_Y, PidController &pidP_X, msr::airlib::MultirotorRpcLibClient &client)
{
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	Vec4i _UpRight;//x,y,a,b
	cv::Mat3b image_bellow = img2mat.get_below_mat();//得到底部图
	std::vector<Vec2i> plat_location;// output
	detect_squares_center(image_bellow, plat_location);

	if (plat_location.size() > 0)
	{
		float py_location = plat_location[0][0];
		float px_location = plat_location[0][1];
		cout << "\nthe pillar is in the scene\n";
		Point XY;
		XY.y = px_location;
		XY.x = py_location;
		cv::circle(image_bellow, XY, 4, cv::Scalar(0, 0, 255));
			//save the picture
			/////如果中心吻合，退出
		if (abs(px_location - 240) < 30 && abs(py_location - 280) < 30)
		{
			count_stop++;
			if (count_stop > 2)
			{
				std::cout << "\n the pillar is recognized\n";
				return true;
			}
		}
		else
		{
			////////// 调整使无人机正对停机坪位置
			std::cout << "pixel...." << std::endl;
			float roll = py_location - 280;
			float pitch = px_location - 240;
			std::cout << "x_pixel: " << pitch << "  y_pixel: " << roll << std::endl;
			slopefly(client, 1, pitch, roll);
			client.hover();
			return false;
		}
	}
	else
	{
		std::cout << "\n the pillar wasn't recognized\n";
		return true;
	}


}
//手标圈位置函数
struct ImageInteractHelper
{
	int start(std::string file) {
		using namespace cv;
		using namespace std;

		int multiple = 1;//图片的放大倍数
		Mat inputImage = imread(file);//这里放置自己的文件路径。
		Mat outputImage;
		resize(inputImage, inputImage, Size(multiple  * inputImage.cols, multiple  * inputImage.rows));
		cvtColor(inputImage, outputImage, COLOR_BGR2HSV);

		//设置鼠标操作回调函数
		namedWindow(WINDOW_NAME);
		setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&outputImage);
		imshow(WINDOW_NAME, inputImage);
		while (1)
		{
			if (waitKey(10) == 27) break;//按下ESC键，程序退出
		}
		waitKey();
		return 0;
	}
}image_interact_helper;
//二维码识别函数
//线性拉伸子函数
void contrastStretch2(cv::Mat &srcImage)
{
	if (srcImage.empty()) {
		std::cerr << "image empty" << std::endl;
		return;
	}
	// 计算图像的最大最小值
	double pixMin, pixMax;
	cv::minMaxLoc(srcImage, &pixMin, &pixMax);
	std::cout << "min_a=" << pixMin << " max_b=" << pixMax << std::endl;

	//create lut table
	cv::Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; i++) {
		if (i < pixMin) lut.at<uchar>(i) = 0;
		else if (i > pixMax) lut.at<uchar>(i) = 255;
		else lut.at<uchar>(i) = static_cast<uchar>(255.0*(i - pixMin) / (pixMax - pixMin) + 0.5);
	}
	//apply lut
	LUT(srcImage, lut, srcImage);
}
//gamma变换子函数
cv::Mat gammaConversion(cv::Mat imgCode) {
	Mat imageGamma(imgCode.size(), CV_32FC3);
	for (int i = 0; i < imgCode.rows; i++) {
		for (int j = 0; j < imgCode.cols; j++) {
			imageGamma.at<Vec3f>(i, j)[0] = (imgCode.at<Vec3b>(i, j)[0])*(imgCode.at<Vec3b>(i, j)[0])*(imgCode.at<Vec3b>(i, j)[0]);
			imageGamma.at<Vec3f>(i, j)[1] = (imgCode.at<Vec3b>(i, j)[1])*(imgCode.at<Vec3b>(i, j)[1])*(imgCode.at<Vec3b>(i, j)[1]);
			imageGamma.at<Vec3f>(i, j)[2] = (imgCode.at<Vec3b>(i, j)[2])*(imgCode.at<Vec3b>(i, j)[2])*(imgCode.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255  
	normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX);
	//转换成8bit图像显示  
	convertScaleAbs(imageGamma, imageGamma);
	return imageGamma;
}
//二维码识别主函数
bool QRdetection(ImageToMat &img2mat, cv::Ptr<aruco::Dictionary> dictionary, Ptr<aruco::DetectorParameters> detectorParams,
	int* ArucoID, int* result, int &countResult) {
	cv::Mat imgCode = img2mat.get_front_mat();
	std::vector< int > ids;
	std::vector<std::vector< Point2f > > corners, rejected;

	//detect markers and estimate pose  先用原图检测，原图过曝，对比度低，可能检测不到，如果检测不到用gamma变换增加对比度后去检测

	aruco::detectMarkers(imgCode, dictionary, corners, ids, detectorParams, rejected);
	if (ids.size() > 0) {
		// draw the code
		aruco::drawDetectedMarkers(imgCode, corners, ids);
		//cv::imshow("src", imgCode);
		//cv::waitKey(100000);
		for (int ids_iter = 0; ids_iter < ids.size(); ids_iter++) {
			for (int aruco_iter = 0; aruco_iter < 10; aruco_iter++) {
				if (ids.at(ids_iter) == ArucoID[aruco_iter]) {
					if (result[aruco_iter] == 1) {
						std::vector<ImageCaptureBase::ImageResponse> responseAruco = img2mat.get_depth_mat();
						SaveResult(imgCode, responseAruco.at(0), corners.at(ids_iter), ArucoID[aruco_iter]);
						result[aruco_iter] = 0;
						countResult++;
						break;
					}
				}
			}
		}
		return true;
	}

	cv::Mat imgGamma = gammaConversion(imgCode);
	aruco::detectMarkers(imgGamma, dictionary, corners, ids, detectorParams, rejected);
	if (ids.size() > 0) {
		// draw the code
		aruco::drawDetectedMarkers(imgCode, corners, ids);
		//cv::imshow("gamma", imgCode);
		//cv::waitKey(100000);
		for (int ids_iter = 0; ids_iter < ids.size(); ids_iter++) {
			for (int aruco_iter = 0; aruco_iter < 10; aruco_iter++) {
				if (ids.at(ids_iter) == ArucoID[aruco_iter]) {
					if (result[aruco_iter] == 1) {
						std::vector<ImageCaptureBase::ImageResponse> responseAruco = img2mat.get_depth_mat();
						SaveResult(imgCode, responseAruco.at(0), corners.at(ids_iter), ArucoID[aruco_iter]);
						result[aruco_iter] = 0;
						countResult++;
						break;
					}
				}
			}
		}
		return true;
	}

	cv::Mat srcGray;
	cvtColor(imgCode, srcGray, CV_BGR2GRAY);
	cv::resize(srcGray, srcGray, cv::Size(), 0.5, 0.5);
	contrastStretch2(srcGray);
	aruco::detectMarkers(srcGray, dictionary, corners, ids, detectorParams, rejected);
	if (ids.size() > 0) {
		// draw the code
		aruco::drawDetectedMarkers(imgCode, corners, ids);
		//cv::imshow("srcGray", srcGray);
		//cv::waitKey(100000);
		for (int ids_iter = 0; ids_iter < ids.size(); ids_iter++) {
			for (int aruco_iter = 0; aruco_iter < 10; aruco_iter++) {
				if (ids.at(ids_iter) == ArucoID[aruco_iter]) {
					if (result[aruco_iter] == 1) {
						std::vector<ImageCaptureBase::ImageResponse> responseAruco = img2mat.get_depth_mat();
						SaveResult(imgCode, responseAruco.at(0), corners.at(ids_iter), ArucoID[aruco_iter]);
						result[aruco_iter] = 0;
						countResult++;
						break;
					}
				}
			}
		}
		return true;
	}


	return false;
}



////////////////////////钻圈补充函数
//三维移动函数
bool front_depth_circle(ImageToMat &img2mat, int &count_go, int &flag_height, int &generation, msr::airlib::MultirotorRpcLibClient &client, PidController &pidP_Z, PidController &pidP_Y)
{
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	Vec4i _UpRight;//x,y,a,b
	cv::Mat img, image_depth, image_depth_C1;
	cv::Mat3b image_front = img2mat.get_front_mat();
	//读深度图，并二值化
	clock_t time_begin = clock();
	std::vector<ImageResponse> response = img2mat.get_depth_mat();
	image_depth_C1 = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC1);
	image_depth = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC3);
	memcpy(image_depth_C1.data, response.at(0).image_data_float.data(), sizeof(float)*response.at(0).height*response.at(0).width);			//img = cv::imdecode(response.at(0).image_data_float, 1);
	image_depth_C1.convertTo(image_depth_C1, CV_32FC1, 1 / 255.0);
	image_depth = convertTo3Channels(image_depth_C1);
	clock_t time_end = clock();
	///检测深度图/场景图中的椭圆
	std::vector<ell::Ellipse> ellipse_rgb = OnImage(image_front);
	//std::vector<ell::Ellipse> ellipse_depth = OnImage(image_depth);
	//std::vector<Vec2i> vec = detect_num(image_front, mats);
	std::cout << "\nthe front result\t" << ellipse_rgb.size() << endl;
	float Max_distance = 0;
	
	for (int iter_num = 1; iter_num < 640*480-1; iter_num++)
	{

		if (response.at(0).image_data_float.at(iter_num) == 255) continue;
		if (Max_distance < response.at(0).image_data_float.at(iter_num) &&
			response.at(0).image_data_float.at(iter_num) <= 5)
			Max_distance = response.at(0).image_data_float.at(iter_num);

	}
	cout << "*****************************" << endl;
	cout << "\n the time cost is  " << time_end - time_begin << endl;
	///前置图里是否有椭圆
	if (ellipse_rgb.size() > 0)
	{
		std::cout << "\nthe circle is in the scene, the distance is in 5m\n";
		int id_maxcircle = 0;
		float Sum_x = 0, Sum_y = 0;
		float a = 0, b = 0;
		//寻找到最大的圈
		for (int i = 0; i < ellipse_rgb.size(); i++)
		{
			if ((0.3*a + 0.7*b) < (0.3*ellipse_rgb[i]._a +0.7*ellipse_rgb[i]._b))//这里进行了加权，但未测试
			{
				a = ellipse_rgb[i]._a;
				b = ellipse_rgb[i]._b;
				id_maxcircle = i;
			}
			Sum_x = ellipse_rgb[id_maxcircle]._xc;
			Sum_y = ellipse_rgb[id_maxcircle]._yc;
		}
		_UpRight[0] = Sum_y;
		_UpRight[1] = Sum_x;
		//计算最大深度
		Max_distance = 0;
		float ThresholdD = 4;
		for (int iter_num = _UpRight[0] * 640; iter_num < _UpRight[0] * 640 + 480; iter_num++)
		{

			if (response.at(0).image_data_float.at(iter_num) == 255) continue;
			if (Max_distance < response.at(0).image_data_float.at(iter_num) &&
				response.at(0).image_data_float.at(iter_num) <= 5)
				Max_distance = response.at(0).image_data_float.at(iter_num);

		}
		std::cout << "\n cout_go \t" << count_go << endl;
		std::cout << "\n Max_distance \t" << Max_distance << endl;
		//判断深度
		if (Max_distance > 3.5)//前进
		{
			client.moveByAngleThrottle(-0.1, 0.0, 0.588, 0.0f, 0.05f);
			std::this_thread::sleep_for(std::chrono::duration<double>(0.05f));
		}
		else if (Max_distance < 2 && Max_distance > 0)//后退
		{
			client.moveByAngleThrottle(0.1, 0.0, 0.588, 0.0f, 0.05f);
			std::this_thread::sleep_for(std::chrono::duration<double>(0.05f));
		}
		else if (Max_distance == 0)//前进
		{
			client.moveByAngleThrottle(-0.1, 0.0, 0.588, 0.0f, 0.08f);
			std::this_thread::sleep_for(std::chrono::duration<double>(0.08f));
		}
		/////如果中心吻合，则直接冲过去
		if (abs(_UpRight[0] - 260) < 10 && abs(_UpRight[1] - 310) < 10 && Max_distance <= 3.5 && Max_distance > 2)
		{
			flag_height = flag_height + 1;
			if (flag_height == 31) flag_height = 0;
			generation = 0;
			return true;
		}
		else if (abs(_UpRight[0] - 260) > 10 || abs(_UpRight[1] - 320) > 10)
		{
			////////// 调整使无人机正对圆心位置 0是H，1是Y，没有setpoint（为啥，以及意义确定）
			std::cout << "\n...........depth..............the location" << std::endl;
			//分段调整
			int sign_th = 0;
			float gravity_d = 0.589;
			float time_f = 0.03;
			if (_UpRight[0] <= 260 - 30)
			{
				sign_th = 1;
				gravity_d = 0.9;
				time_f = 0.15;
			}
			else if ((_UpRight[0] <= 260 - 3) && (_UpRight[0] > 260 - 30))
			{
				sign_th = 2;
				gravity_d = 0.6;
				time_f = 0.1;
			}
			else
			{
				sign_th = 3;
				gravity_d = 0.589;
			}
			float delta_throttle = (pidP_Z.control(_UpRight[0] - 20)) + gravity_d;
			float roll = pidP_Y.control(_UpRight[1] );
			std::cout << "H_pixel: " << _UpRight[0] - 260 << "  \nY_pixel: " << _UpRight[1] - 320 << std::endl;
			CLIP3(-0.3, roll, 0.3);
			CLIP3(0.3, delta_throttle, 0.80);//注意这里为了稳定将0.85降到了0.80
			client.moveByAngleThrottle(0, -roll, delta_throttle, 0.0f, time_f);
			std::this_thread::sleep_for(std::chrono::duration<double>(time_f));
			std::cout << "roll: " << roll << "  delta_throttle: " << delta_throttle << std::endl;
			return true;
		}
		return true;
	}
	else if (ellipse_rgb.size() <= 0)
	{
		std::cout << "\n the max distance is " << Max_distance;
		if (Max_distance < 3 && Max_distance > 0)//后退
		{
			client.moveByAngleThrottle(0.1, 0.0, 0.588, 0.0f, 0.02f);
			std::this_thread::sleep_for(std::chrono::duration<double>(0.04f));
		}
		else if (Max_distance == 0)
		{
			generation++;
			if (generation > 2)
			{
				flag_height = flag_height + 1;
				if (flag_height == 31) flag_height = 0;
				generation = 0;
				return false;
			}
		}
		
	}

}

void circle_test(ImageToMat &img2mat, int &count_go, int &flag_height, int &generation, msr::airlib::MultirotorRpcLibClient &client, PidController &pidP_Z, PidController &pidP_Y)
{
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	Vec4i _UpRight;//x,y,a,b
	cv::Mat img, image_depth, image_depth_C1;
	cv::Mat3b image_front = img2mat.get_front_mat();
	//读深度图，并二值化
	std::vector<ImageResponse> response = img2mat.get_depth_mat();
	image_depth_C1 = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC1);
	image_depth = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC3);
	std::cout << "response.at(0).height: " << response.at(0).height << " response.at(0).width" << response.at(0).width << std::endl;
	memcpy(image_depth_C1.data, response.at(0).image_data_float.data(), sizeof(float)*response.at(0).height*response.at(0).width);			//img = cv::imdecode(response.at(0).image_data_float, 1);
	image_depth_C1.convertTo(image_depth_C1, CV_32FC1, 1 / 255.0);
	image_depth = convertTo3Channels(image_depth_C1);

	///检测深度图/场景图中的椭圆
	//std::vector<ell::Ellipse> ellipse_rgb = OnImage(image_front);
	std::vector<ell::Ellipse> ellipse_depth = OnImage(image_depth);
	//std::vector<Vec2i> vec = detect_num(image_front, mats);
	std::cout << "\nthe test result\t" << ellipse_depth.size() << endl;
	///深度图里是否有椭圆
	if (ellipse_depth.size() > 0)
	{
		//DrawDetectedEllipses(image_depth, ellipse_depth);
		std::cout << "\nthe circle is in the scene\n";
		int id_maxcircle = 0;
		float Sum_x = 0, Sum_y = 0;
		float a = 0, b = 0;
		//寻找到最大圈
		for (int i = 0; i < ellipse_depth.size(); i++)
		{
			if ((0.3*a + 0.7*b) < (0.3*ellipse_depth[i]._a +0.7* ellipse_depth[i]._b))//这里进行了加权，但未测试
			{
				a = ellipse_depth[i]._a;
				b = ellipse_depth[i]._b;
				id_maxcircle = i;
			}
			Sum_x = ellipse_depth[id_maxcircle]._xc;
			Sum_y = ellipse_depth[id_maxcircle]._yc;
		}
		_UpRight[0] = Sum_y;
		_UpRight[1] = Sum_x;

		/////如果中心吻合，则直接冲过去
		if (abs(_UpRight[0] - 260) < 10 && abs(_UpRight[1] - 320) < 10)
		{
			//距离足够近，则快速飞过去  参数可能还需调整
			int height, width;
			height = _UpRight[0] + 0.5*(a + b);
			width = _UpRight[1] + 0.5*(a + b);
			if (height > 480) height = 480;
			if (width > 640) width = 640;
			xstay(client, 0.5);//track
			float Max_distance = 0;
			float ThresholdD = 4;
			for (int iter_num = 1; iter_num < _UpRight[0] * 640 + _UpRight[1]; iter_num++)
			{

				if (response.at(0).image_data_float.at(iter_num) == 255) continue;
				if (Max_distance < response.at(0).image_data_float.at(iter_num) &&
					response.at(0).image_data_float.at(iter_num) <= 5)
					Max_distance = response.at(0).image_data_float.at(iter_num);

			}
			std::cout << "\n cout_go \t" << count_go << endl;
			std::cout << "\n Max_distance \t" << Max_distance << endl;
			count_go++;
			if (count_go > 2)
			{
				std::cout << "gogogogoigoo....." << std::endl;
				if (Max_distance >= 4.0)//判断距离圈的距离来决定走得距离
				{
					client.moveByAngleThrottle(-0.1, 0, 0.589, 0.0f, 2.9f);//2.9需要改动
					std::this_thread::sleep_for(std::chrono::duration<double>(2.9f));
				}
				else
				{
					client.moveByAngleThrottle(-0.1, 0, 0.589, 0.0f, 2.6f);//the length is up to the distance
					std::this_thread::sleep_for(std::chrono::duration<double>(2.6f));
				}
				client.hover();
				std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
				count_go = 0;
			}
		}
		else
		{
			////////// 调整使无人机正对圆心位置 0是H，1是Y，没有setpoint（为啥，以及意义确定）
			std::cout << "\n...........depth..............the location" << std::endl;
			//分段调整
			int sign_th = 0;
			float gravity_d = 0.589;
			float time_f = 0.1;
			float roll;
			if (_UpRight[0] <= 260 - 30)
			{
				sign_th = 1;
				gravity_d = 0.593;
				time_f = 0.15;
			}
			else if ((_UpRight[0] <= 260 - 3) && (_UpRight[0] > 260 - 30))
			{
				sign_th = 2;
				gravity_d = 0.590;
				time_f = 0.10;
			}
			else
			{
				sign_th = 3;
				gravity_d = 0.589;
			}

			float delta_throttle = (pidP_Z.control(_UpRight[0] -20)) + gravity_d;
			roll = pidP_Y.control(_UpRight[1] + 10);
			std::cout << "H_pixel: " << _UpRight[0] - 260 << "  \nY_pixel: " << _UpRight[1] - 320 << std::endl;
			cout << "\n the roll is" << roll << endl;
			CLIP3(0.3, delta_throttle, 0.85);
			CLIP3(-0.2, roll, 0.2);
			client.moveByAngleThrottle(0, -roll, delta_throttle, 0.0f, time_f);
			std::this_thread::sleep_for(std::chrono::duration<double>(time_f));
			std::cout << "roll: " << roll << "  delta_throttle: " << delta_throttle << std::endl;
		}
	}
	else
	{
		generation++;
		if (generation > 1)
		{
			flag_height = flag_height + 1;
			if (flag_height == 31) flag_height = 0;
			generation = 0;
		}
	}

}
// 显示前置与底部相机图函数，并输出结果
bool front_circle(ImageToMat &img2mat, int &count_go, int &flag_height, int &generation, msr::airlib::MultirotorRpcLibClient &client, PidController &pidP_Z, PidController &pidP_Y)
{
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	Vec4i _UpRight;//x,y,a,b
	cv::Mat img, image_depth, image_depth_C1;
	cv::Mat3b image_front = img2mat.get_front_mat();

	///检测深度图/场景图中的椭圆
	std::vector<ell::Ellipse> ellipse_rgb = OnImage(image_front);
	std::cout << "\nthe test result\t" << ellipse_rgb.size() << endl;


	///图里是否有椭圆
	if (ellipse_rgb.size() > 0)
	{
		//DrawDetectedEllipses(image_depth, ellipse_depth);
		std::cout << "\nthe circle is in the scene\n";
		//cv::namedWindow("imageScene");
		//cv::imshow("imageScene", image_front);
		//cv::waitKey(1);
		int id_maxcircle = 0;
		float Sum_x = 0, Sum_y = 0;
		float a = 0, b = 0;
		a = ellipse_rgb[0]._a;
		b = ellipse_rgb[0]._b;
		///多个取均值
		for (int i = 0; i < ellipse_rgb.size(); i++)
		{
			if ((0.3*a + 0.7*b) < (0.3*ellipse_rgb[i]._a + 0.7*ellipse_rgb[i]._b))
			{
				a = ellipse_rgb[i]._a;
				b = ellipse_rgb[i]._b;
				id_maxcircle = i;
			}
			Sum_x = ellipse_rgb[id_maxcircle]._xc;
			Sum_y = ellipse_rgb[id_maxcircle]._yc;
		}
		_UpRight[0] = Sum_y;
		_UpRight[1] = Sum_x;

		/////如果中心吻合，退出
		if (abs(_UpRight[0] - 260) < 10 && abs(_UpRight[1] - 310) < 10)
		{
			flag_height = flag_height + 1;
			if (flag_height == 31) flag_height = 0;
			generation = 0;
			return true;
		}
		else
		{
			////////// 调整使无人机正对圆心位置 0是H，1是Y，没有setpoint（为啥，以及意义确定）
			std::cout << "\n..........front...............the location" << std::endl;
			//分段调整
			int sign_th = 0;
			float gravity_d = 0.589;
			if (_UpRight[0] <= 260 - 30)
			{
				sign_th = 1;
				gravity_d = 0.6;
			}
			else if ((_UpRight[0] <= 260 - 3) && (_UpRight[0] > 260 - 30))
			{
				sign_th = 2;
				gravity_d = 0.591;
			}
			else
			{
				sign_th = 3;
				gravity_d = 0.589;
			}
			float delta_throttle = (pidP_Z.control(_UpRight[0] -20)) + gravity_d;
			float roll = pidP_Y.control(_UpRight[1] + 10);
			std::cout << "H_pixel: " << _UpRight[0] - 260 << "  \nY_pixel: " << _UpRight[1] - 310 << std::endl;
			CLIP3(-0.3, roll, 0.3);
			CLIP3(0.3, delta_throttle, 0.9);
			client.moveByAngleThrottle(0, -roll, delta_throttle, 0.0f, 0.1f);
			std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
			std::cout << "roll: " << roll << "  delta_throttle: " << delta_throttle << std::endl;
			return true;
		}
		return true;
	}
	else
	{
		generation++;
		if (generation > 3)
		{
			flag_height = flag_height + 1;
			if (flag_height == 31) flag_height = 0;
			generation = 0;
		}
		return false;
	}

}
// 显示前置与底部相机图函数，并输出结果
void circle_imshow(ImageToMat &img2mat)
{
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	Vec4i _UpRight;//x,y,a,b
	cv::Mat img, image_depth, image_depth_C1;
	cv::Mat3b image_front = img2mat.get_front_mat();
	//读深度图，并二值化
	std::vector<ImageResponse> response = img2mat.get_depth_mat();
	image_depth_C1 = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC1);
	image_depth = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC3);
	std::cout << "response.at(0).height: " << response.at(0).height << " response.at(0).width" << response.at(0).width << std::endl;
	memcpy(image_depth_C1.data, response.at(0).image_data_float.data(), sizeof(float)*response.at(0).height*response.at(0).width);			//img = cv::imdecode(response.at(0).image_data_float, 1);
	image_depth_C1.convertTo(image_depth_C1, CV_32FC1, 1 / 255.0);
	image_depth = convertTo3Channels(image_depth_C1);

	///检测深度图/场景图中的椭圆
	std::vector<ell::Ellipse> ellipse_rgb = OnImage(image_front);
	std::vector<ell::Ellipse> ellipse_depth = OnImage(image_depth);
	//std::vector<Vec2i> vec = detect_num(image_front, mats);
	cout << "\nthe image_front result\t" << ellipse_rgb.size() << endl;
	cout << "\nthe ellipse_depth result\t" << ellipse_depth.size() << endl;
	//cv::namedWindow("image_front");
	//cv::imshow("image_front", image_front);
	//cv::waitKey(1);
	//cv::namedWindow("ellipse_depth");
	//cv::imshow("ellipse_depth", image_depth);
	//cv::waitKey(1);

}
int ori_heightcontrol(float num, prev_states* prev,msr::airlib::MultirotorRpcLibClient &client, BarometerData &Barometer_origin, PidController &pidZ)
{
	float gravity_d=0.6;// 平衡量
	int stay = 0;// 保证稳定停止量
	while (1) {
		cout << "\n the test control mode\n";
		////the Begin action
		BarometerData curr_bardata = client.getBarometerdata();//气压计数据
		float cur_height = curr_bardata.altitude - Barometer_origin.altitude;//世界坐标系
		cur_height = kalman(cur_height, prev, 0, Q, R);//滤波 the height
		std::cout << "ned_curr: " << cur_height << std::endl;
		float delta_throttle = pidZ.control(cur_height) + 0.6;//control the A
		CLIP3(0.4, delta_throttle, 0.8);//取中间值
		client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);// THE mid clent
		std::this_thread::sleep_for(std::chrono::milliseconds(10));// stop the other action
		 // the height fly
		pidZ.setPoint(num, 0.3, 0, 0.4);
		// judge the flow direction
		if (num > cur_height)
			gravity_d = 0.587;
		else
			gravity_d = 0.58;
		delta_throttle = pidZ.control(cur_height) + gravity_d;
		CLIP3(0.4, delta_throttle, 0.8);//取中间值
		client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.2f);// 
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		// stop the action
		if ((num - cur_height) < 0.1 && (num - cur_height) > -0.1)	
		{
			cout << "\naim is ready\n";
			client.hover();
			xstay(client,0.5);
			client.hover();
			return 1;
		}
	}
	
}

// 获得深度图以及前置相机图片
//路径
//int to string
std::string transfer_int(int num)
{
	std::stringstream ss;
	ss << num;
	std::string s1 = ss.str();
	std::string s2;
	ss >> s2;
	return s2;
}
std::string img_dir_path = "D:/ROS/Competition0730/picture_get/1/";
void get_img(ImageToMat &img2mat,int &generation)
{
    
    typedef ImageCaptureBase::ImageResponse ImageResponse;
	Vec4i _UpRight;//x,y,a,b
	cv::Mat img, image_depth, image_depth_C1;
	cv::Mat3b image_front = img2mat.get_front_mat();
	//读深度图，并二值化
	std::vector<ImageResponse> response = img2mat.get_depth_mat();
	image_depth_C1 = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC1);
	image_depth = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC3);
	std::cout << "response.at(0).height: " << response.at(0).height << " response.at(0).width" << response.at(0).width << std::endl;
	memcpy(image_depth_C1.data, response.at(0).image_data_float.data(), sizeof(float)*response.at(0).height*response.at(0).width);			//img = cv::imdecode(response.at(0).image_data_float, 1);
	image_depth_C1.convertTo(image_depth_C1, CV_32FC1, 1 / 255.0);
	image_depth = convertTo3Channels(image_depth_C1);
    //获得保存路径与名称
    std::string depth_path, front_path,number_picture;
	number_picture = transfer_int(generation);
    depth_path = img_dir_path + "depth/" + number_picture + ".jpg";
    front_path = img_dir_path + "front/" + number_picture + ".jpg";
    cv::imwrite(depth_path,image_depth);
    cv::imwrite(front_path,image_front);
    generation = generation + 1;
}

int main()//begin
{
	using namespace std;
	using namespace msr::airlib;

	cv::Mat img, image_depth, image_depth_C1;
	msr::airlib::MultirotorRpcLibClient client;
	ImageToMat img2mat;
	typedef ImageCaptureBase::ImageRequest ImageRequest;
	typedef ImageCaptureBase::ImageResponse ImageResponse;
	typedef ImageCaptureBase::ImageType ImageType;
	typedef common_utils::FileSystem FileSystem;

	try
	{
		client.confirmConnection();
		client.enableApiControl(true);
		client.armDisarm(true);		
		BarometerData curr_bardata;
		ImuData Imudata;
		MagnetometerData Target_Magdata, Magdata;
		Target_Magdata = client.getMagnetometerdata();
		float target_Mag_y = -Target_Magdata.magnetic_field_body.y();
		Vector3r ned_origin, ned_curr, ned_target, home_ned, control_origin;
		Vector3r ArucoBegin;
		BarometerData Barometer_origin = client.getBarometerdata();
		BarometerData point_control_bardata;

		int count0 = 0, count1 = 0, count2 = 0, count3 = 0, count4 = 0, count_circle = 0, count_parking = 0, count_parking_1 = 0, count_left = 0, count_right = 0;
		int count_parking_2 = 0;
		int count_parking_local = 0;
		int count_home = 0;
		int count_code = 0, count_yaw = 0;
		int count_code_alt = 0;
		int currentnumber = 0, nextnumber = 10;
		int TREE_NUM = 0;
		int UPDOWN_COUNT = 0;
		int controlmode = 4;// control the begin
		int i_kalman = 0; //for kalman z
		int x_kalman = 0;
		int y_kalman = 0;
		int TREE_COUNT = 0;
		int HOME_COUNT = 0;
		int flag_con;
		int flag_dis;
		int Clloctnum = 40;
		float test_delta_pitch;
		float theta, target_theta;
		bool flag = false; //for circle
		bool flag_parking = false;
		bool parking = false;
		bool flag_image = false;
		cv::Mat image, Img;
		Vec2i xy, xy_temp,XY_TREE;
		Vec2f radius;
		prev_states prev[1] = { 0,0 };
		prev_states prev_x[1] = { 0,0 };
		prev_states prev_y[1] = { 0,0 };
		bool flag_mode3 = false;
		bool isFirst = false;
		bool ReadTxt = false;	
		bool FLAG_CB = false;
		bool turnforward = false;
		char traindataPath[256];
		//////// target position ////////////////	
		ned_target(2) = 24;//7 the height
		ned_target(0) = 240;//这里认为图像的纵向为x轴，为了与无人机保持一致
		ned_target(1) = 320;
		//////////pidcontroller////////////////////
		PidController pidX, pidY, pidZ, pidP_X, pidP_Y, pidP_Z, pid_yaw;
		pidP_X.setPoint(ned_target(0), 0.0015, 0, 0.0005);// 这里的x指的是以无人机运动方向为x的反方向
		pidP_Y.setPoint(ned_target(1), 0.0013, 0, 0.0005);
		pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
		pidP_Z.setPoint(ned_target(0), 0.05, 0, 0.0001);
		pid_yaw.setPoint(target_Mag_y, 2, 0, 1);
		flag_con = 1;
		flag_dis = 1;
		std::cout << "Press Enter to arm the drone." << std::endl;
		/************************** for collect data  ******************************/
		char filename[100];
		char dataPath[256];
		int data_i = 0;
		
		bool flag_collect_data = false;
		bool circle_middle_flag = false, circle_far7_flag = false, circle_far10_flag = false,
			circle_far12_flag = false, circle_far13_flag = false, circle_far16_flag = false, circle_far18_flag = false, circle_far14_flag=false;
		bool last_circle = false;
		bool flag_parking_10 = false;
		bool flag_num10 = false;
		bool FLAG_UPDOWN = true;
		bool is_FLAG_UP = false;
		bool is_FLAG_DOWN = false;
		bool HOME = false;
	 
		int count_num10 = 0;
		int count_collect_data = 0;
		float last_circle_weight = 1;
		float Min_distance = 0;
		std::vector<cv::Mat> ImageForPosition;
		std::vector<cv::Mat> ImageForPosition_back;
		std::vector<Vec3f> LocalPosition;
		int posarray[10] = { 0,0,0,1,1,2,2,1,2,2 };//用来存放位置数值
		int doubt_array[10] = { 0,1,1,1,1,1,1,1,1,1 };//用来存放怀疑数组，0-不怀疑，1-怀疑
		int predict_check_position[10] = { 0 };//用来存放预测+check数组;
	    //int col_4[10] = { 448,448,448,448,488,488,448,448,334,448 };
		/*int map_pic_split[10][2] = {
			204,448,//circle 1
			170,470,//circle 2
			204,448,//circle 3
			204,448,//circle 4
			204,484,//circle 5
			204,484,//circle 6
			204,448,//circle 7
			204,448,//circle 8
			204,334,//circle 9
			170,470//circle 10
		};*/

		
		//float realdelta_x[10] = { 11.6,12.5,10.1,11.1,12.7,10.1,16.7,15.6,11.4,13.5 };//用来记录无人机大致拍照的delta位置
		//float realdelta_x[10] = { 19.6,14.5,13.1,11.9,15.4,13.4,16.7,15.6,13.4,15.3 };//用来记录无人机大致拍照的delta位置
		float realdelta_x[10] = { 19.1,14.5,13.1,11.9,15.4,13.4,16.7,15.3,13.7,15.7 };//用来记录无人机大致拍照的delta位置
		float location_x[10] = { 2.0, 4.74, 3.1, 2.34, 6.0, 3.1, 7.054, 7.15, 2.67, 5.2 };//13.5
	//float location_x[10] = { 2.0, 7.47, 6.578, 5.34, 10.0, 6.96, 10.544, 10.65, 5.67, 8.5 };//13.5
	//{ 10.5, 7.47, 6.578, 5.34, 10.47, 6.86, 10.544, 10.65, 5.62, 8.5 };
   //float location_x[10] = { 10.5, 7.47, 6.578, 5.34, 10.50, 6.96, 10.544, 10.65, 5.67, 8.5 };//13.5
		float location_y[10][3] =
		{ 11.97, 11.97, 11.97,//1

			11.43, -0.17, -11.5,//2

			7.94, -4.54, -20.69,//3

			9.71, -5.85, -20.69,//4

			7.93, -12.92, -28.6,//5

			8.23, -13.0, -31.93,//6

			8.23, -10.62, -24.23,//700

			5.97, -8.33, -26.2,//8

			11.5, -6.5, -16.0,//9  有问题，建议重新标定

			16.0, -5, -20.88 };//10
		//////////////////////////// for testing controller  /////////////////////
	    std::vector<Vec3f> LocalPosition_temp;
		Vec3f distance;

		//////ARuco

		int iter_num;
		int rollCoefficient, pitchCoefficient;
		bool statechange = false;
		bool flag_yaw = false;
		int ArucoID[5];
		int result[5] = { 1,1,1,1,1 };
		int countResult = 0;

		ifstream in;
		in.open("D:\\aruco.txt", ios::in);
		int iterAruco = 0;
		while (!in.eof() && iterAruco < 5)
		{
			in >> ArucoID[iterAruco];
			iterAruco++;
		}
		in.close();

		
		Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);
		Mat out;
		dictionary->drawMarker(100, 600, out, 5);
		Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
		detectorParams->adaptiveThreshWinSizeMin = 3;
		detectorParams->adaptiveThreshWinSizeMax = 3;
		detectorParams->adaptiveThreshWinSizeStep = 1;
		detectorParams->minDistanceToBorder = 0;
		detectorParams->polygonalApproxAccuracyRate = 0.15;
		detectorParams->minMarkerPerimeterRate = 0.15;
		typedef enum enumType
		{
			right2left,
			forward,
			left2right,
		};
		enumType state = right2left;
        ////controlmode12  data
		int count_go = 1;// 用于中心稳定计数
		int flag_height = 0;//用于判断搜索高度
		int generation = 0;
		int motion_i = 0;//用于圈个数计数
		data_circle circle_1;
		std::vector <data_circle> data_cirall;
		data_circle pillar_1;
		std::queue <data_circle> data_pilall;
		pillar_1.start(6.3, 12.85, 4.4);//1   1号柱子 pillar_1  3.6向前冲太多 3.0冲太少看不清
		data_pilall.push(pillar_1);
		pillar_1.start(1, -18.387, 4.4);//2  2号柱子
		data_pilall.push(pillar_1);
		pillar_1.start(11.3, -16, 6.181);//6号柱子 暂时未测
		data_pilall.push(pillar_1);
		pillar_1.start(4.0, 13.5, 5.5);//5号柱子
		data_pilall.push(pillar_1);
		pillar_1.start(-2.5, 10.0, 6.181);//4号柱子
		data_pilall.push(pillar_1);
		pillar_1.start(7.5, 10.5, 6);//3
		data_pilall.push(pillar_1);
		pillar_1.start(13.0, 8.0, 5);//7
		data_pilall.push(pillar_1);
		pillar_1.start(1, -10, 5);//8
		data_pilall.push(pillar_1);
		pillar_1.start(1, -12.0, 5);//9
		data_pilall.push(pillar_1);
		pillar_1.start(3.5, -12, 5);//10
		data_pilall.push(pillar_1);
		bool ifneedtodown = false;//用于控制是否识别此柱子
		//数据结构定义
		//钻圈图像获取
		//get_img img_circle;
		//img_circle.start(&img2mat);
		//直接右移数据
		data_circle pillar_2;
		std::queue <data_circle> data_pilright;
		//高度获得初始化
		get_data hdata;
		hdata.start(prev, &client, Barometer_origin);
        // 这里控制起始进入模式
		controlmode = 0;
		// load template
		std::vector<cv::Mat> template_pic;
		for (int i = 1; i < 10; i++) {  // 9 times
			sprintf(filename, "template\\%d_template.png", i); // TODO
			cv::Mat read_in = cv::imread(filename);
			template_pic.push_back(read_in);
		}


		while (1)
		{
			clock_t begin = clock();
			if (controlmode == 0)  //0起飞
			{
				
				int generation = 0;
                char input;
                while(1)
                {
                    cout<<"##############################################################"<<endl;
                    cout<<"whether get picture or not ,yes is 1,no is other"<<endl;
                    cin>>input;
                    if(input == '1')
                    {
						get_img(img2mat, generation);
                        cout<<"the number of picture is : "<<generation<<endl;
                    }
                }
			}
			else if (controlmode == -1) //the 11_pre-step mode
			{
				data_circle pillar_mid;
				pillar_mid.start(145, -2.5);
				pillar_height(15, prev, client, Barometer_origin, pidZ);
				xcontrol(client, 1, 0.0, pillar_mid.roll, pillar_mid.time_r);//y
				client.hover();
				xstay(client, 1.0);
				client.hover();
				xcontrol(client, 1, pillar_mid.pitch, 0.0, pillar_mid.time_p);//x
				client.hover();
				xstay(client, 1.0);
				client.hover();
				pillar_height(4.1, prev, client, Barometer_origin, pidZ);
				//钻圈
				while (1)
				{
					circle_test(img2mat, count_go, flag_height, generation, client, pidP_Z, pidP_Y);
					xstay(client, 0.4);
					cout << "\n the flag is\t" << flag_height;
					if (count_go == 0) break;
					if (flag_height == 1)
					{
						xcontrol(client, 1, 0.1, 0.0, 3.5);
						flag_height++;
					}
					else if (flag_height == 3)
					{
						xcontrol(client, 1, 0.1, 0.0, 0);
						flag_height++;
					}
					else if (flag_height == 5)
					{
						ori_heightcontrol(4.1, prev, client, Barometer_origin, pidZ);
						xcontrol(client, 1, -0.1, 0.0, 1.4);
						while (flag_height == 5)
						{
							front_circle(img2mat, count_go, flag_height, generation, client, pidP_Z, pidP_Y);
							xstay(client, 0.4);
						}
						xcontrol(client, 1, 0.1, 0.0, 1.4);

					}
					else if (flag_height == 7)
					{
						xcontrol(client, 1, -0.1, 0.0, 1.4);
						flag_height++;
					}
					else if (flag_height == 9)
					{
						xcontrol(client, 1, 0.1, 0.0, 2.2);
						flag_height++;
					}
					else if (flag_height == 11)
					{
						ori_heightcontrol(4.1, prev, client, Barometer_origin, pidZ);
						xcontrol(client, 1, -0.1, 0.0, 1.4);
						client.hover();
						xcontrol(client, 1, 0.0, 0.1, 1.4);
						flag_height++;
					}
					else if (flag_height == 13)
					{
						ori_heightcontrol(4.1, prev, client, Barometer_origin, pidZ);
						xcontrol(client, 1, 0.0, 0.1, 2.2);
						flag_height++;
					}
					else if (flag_height == 15)
					{
						xcontrol(client, 1, 0.0, 0.1, 1.4);
						xcontrol(client, 1, 0.1, 0.0, 2.0);
						break;
						flag_height++;
					}
				}
				controlmode = 11;
			}
			//采图模式
			else if (controlmode == 10)
			{

				curr_bardata = client.getBarometerdata();
				ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
				ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);    //300多行的，三轴坐标，QR去年设好的，kalman滤波估计高度
				std::cout << "ned_curr: " << ned_curr(2) << std::endl;
				float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;      //推力控制量，//控制螺旋桨前后推力
				std::cout << "delta_throttle: " << delta_throttle << std::endl;
				CLIP3(0.4, delta_throttle, 0.8);//取饱和值，取（0.4,0.8之间）
				client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);//变成了阻塞的线程，不科学，应该是多线程并行
				std::this_thread::sleep_for(std::chrono::milliseconds(10));//动一下停一下
				int i;

				if (abs(ned_target(2) - ned_curr(2)) < 0.08)//设定的目标高度小于当前的目标高度0.08，不断让目标靠近
				{
					count4++;
				}
				if (count4 > 15)//有四十次和目标高度很相近
				{
					// first fly right;

					client.moveByAngleThrottle(0.0f, 0.1f, 0.59, 0.0f, 3.50f);//roll向右飞一段，取高度为25m，采样张数为n张
					std::this_thread::sleep_for(std::chrono::milliseconds(3500));//
					client.hover();//悬停，不然下一步转换方向前进不稳
					std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));

					float collect_time = 18;
					int collect_num = 36;

					client.moveByAngleThrottle(-0.1f, 0.0f, 0.592, 0.0f, collect_time);//roll向右飞一段，取高度为25m，采样张数为n张
					//TODO
					for (int i = 0; i < collect_num; i++) {
						image = img2mat.get_front_mat();
						ImageForPosition.push_back(image);
						std::this_thread::sleep_for(std::chrono::milliseconds(500));
					}
					for (int i = 0; i < collect_num; i++) {
						sprintf(filename, "front\\%d-front.png", i);
						imwrite(filename, ImageForPosition.at(i));
					}
					pillar_height(24, prev, client, Barometer_origin, pidZ);

					client.moveByAngleThrottle(0.1f, 0.0f, 0.592, 0.0f, collect_time);//roll向右飞一段，取高度为25m，采样张数为n张

					for (int i = 0; i < collect_num; i++) {
						image = img2mat.get_below_mat();
						ImageForPosition_back.push_back(image);
						std::this_thread::sleep_for(std::chrono::milliseconds(500));
					}
					for (int i = collect_num - 1; i >= 0; i--) {
						sprintf(filename, "below\\%d-below.png", i);
						imwrite(filename, ImageForPosition_back.at(collect_num - 1 - i));
					}
					//pillar_height(24, prev, client, Barometer_origin, pidZ);
					std::reverse(ImageForPosition_back.begin(), ImageForPosition_back.end());
					//向左
					client.moveByAngleThrottle(0.0f, -0.1f, 0.59, 0.0f, 3.3f);
					std::this_thread::sleep_for(std::chrono::milliseconds(3300));
					
					
					//定高
					pillar_height(10, prev, client, Barometer_origin, pidZ);
					//稳定
					client.hover();
					std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));

					controlmode = 40;
				}


				//int photonum = 11;
				//
				//if (count4 > 40)//有四十次和目标高度很相近
				//{
				//	
				//	i = photonum;
				//	for (i = 1; i < photonum; i++)//存图
				//	{
				//		sprintf(filename, "%d.png", i);
				//		flag_collect_data = false;
				//		while (!flag_collect_data)
				//		{
				//			data_circle planego;
				//			curr_bardata = client.getBarometerdata();//定高度，要保证高度相同，才能拼成一整张图，
				//			ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
				//			ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
				//			//pid_zerror << ned_curr(2) << endl;
				//			std::cout << "ned_curr: " << ned_curr(2) << std::endl;
				//			if (abs(ned_target(2) - ned_curr(2)) < 0.2) count_collect_data++;
				//			if (count_collect_data > 20)
				//			{
				//				planego.start(realdelta_x[i - 1], 0);//设置需要的delta_x
				//				xcontrol(client, 1, planego.pitch, 0, planego.time_p);//根据计算出来的移动到需要的位置
				//				client.hover();
				//				xstay(client, 1.0);
				//				client.hover();
				//				clock_t start = clock();
				//				image = img2mat.get_below_mat();
				//				imwrite(filename, image);
				//				ImageForPosition.push_back(image);
				//				flag_collect_data = true;
				//			}
				//			else
				//			{
				//				//调整
				//				float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
				//				std::cout << "delta_throttle: " << delta_throttle << std::endl;
				//				CLIP3(0.4, delta_throttle, 0.8);
				//				client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);
				//				std::this_thread::sleep_for(std::chrono::duration<double>(0.01f));
				//			}
				//		}
				//		count_collect_data = 0;
				//	}

				//	client.hover();
				//}

				//if (i == photonum)//采够n张，
				//{

				//	for (int u = 0; u < 10; u++)
				//	{
				//		cv::Mat pic1, pic2;
				//		pic1 = ImageForPosition.at(u);
				//		pic1.copyTo(pic2);//拷贝pic的数据区到image中
				//		line(pic2, Point(1, 160), Point(640, 160), Scalar(0, 0, 255), 1, CV_AA);
				//		line(pic2, Point(1, 320), Point(640, 320), Scalar(0, 0, 255), 1, CV_AA);
				//		line(pic2, Point(204, 1), Point(204, 480), Scalar(0, 0, 255), 1, CV_AA);
				//		line(pic2, Point(col_4[u], 1), Point(col_4[u], 480), Scalar(0, 0, 255), 1, CV_AA);
				//		std::vector<Vec4i> range;
				//		std::vector<Vec3i> circles = detect_circle(pic2, range, u + 1);//打算在这一步之后直接计算每一张图片rela_distance
				//		cv::Mat binImage_h;
				//		if (u == 0)
				//		{
				//			int res = pos_detect(pic1, binImage_h, u + 1, 0);
				//			posarray[u] = res;//写入判断数组
				//		}
				//		else
				//		{
				//			int res = pos_detect(pic1, binImage_h, u + 1, posarray[u - 1]);
				//			posarray[u] = res;//写入判断数组
				//		}
				//	}
				//	//修正前
				//	cout << "Before being corrected" << endl;
				//	for (int u = 0; u < 10; u++)
				//		std::cout << posarray[u];

				//	std::cout << endl;
				//	predict_check_position[0] = 0;

				//	for (int u = 1; u < 10; u++)
				//	{
				//		int lastpos = posarray[u - 1];
				//		int p = EzMap::DetectCircle_BaseOn_UpDown(ImageForPosition, u, lastpos);
				//		predict_check_position[u] = p;
				//		if (p != posarray[u])
				//		{
				//			doubt_array[u] = 1;
				//		}
				//		if (posarray[u] == -1)
				//		{
				//			doubt_array[u] = 1;
				//			posarray[u] = p;
				//		}
				//	}

				//	//修正后
				//	cout << "posarray" << endl;
				//	for (int u = 0; u < 10; u++)
				//		std::cout << posarray[u];
				//	std::cout << endl;

				//	cout << "predict_check_position" << endl;
				//	for (int u = 0; u < 10; u++)
				//		std::cout << predict_check_position[u];
				//	std::cout << endl;


				//	//准备回到停机坪，以初始姿态
				//	client.moveByAngleThrottle(0.0f, -0.1f, 0.59, 0.0f, 3.00f);//roll向左飞一段，取高度为height(m)，采样张数为n张
				//	std::this_thread::sleep_for(std::chrono::milliseconds(3000));//
				//	client.hover();//悬停，不然下一步转换方向前进不稳
				//	std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));
				//	client.moveByAngleThrottle(0.1f, 0.0f, 0.59, 0.0f, 16.5f);//向后飞
				//	std::this_thread::sleep_for(std::chrono::milliseconds(16500));//这个前进长度刚好,不到一点点
				//	client.hover();//悬停
				//	std::this_thread::sleep_for(std::chrono::duration<double>(3.0f));
				//	pillar_height(10, prev, client, Barometer_origin, pidZ);
				//	client.hover();//悬停
				//	std::this_thread::sleep_for(std::chrono::duration<double>(3.0f));
				//	//以上飞到停机坪附近，应该是不到一些
				//	//pillar_height(12, prev, client, Barometer_origin, pidZ);
				//	//client.hover();
				//	int count_stop = 0;
				//	while (1)
				//	{
				//		if (platform_move_PARK(img2mat, count_stop, pidP_Y, pidP_X, client))
				//		{
				//			std::cout << "满足中心条件" << endl;
				//			break;
				//		}
				//	}

				//	//下降
				//	pillar_height(6.0, prev, client, Barometer_origin, pidZ);
				//	count_stop = 0;
				//	while (1)
				//	{
				//		if (platform_move_PARK(img2mat, count_stop, pidP_Y, pidP_X, client))
				//		{
				//			std::cout << "满足中心条件" << endl;
				//			break;
				//		}
				//	}
				//	/*client.moveByAngleThrottle(0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
				//	std::this_thread::sleep_for(std::chrono::duration<double>(1.0f));*/
				//	
				//}
				//

			}
			else if (controlmode == 40) {
				cout << "mode 40" << endl;
				char filenames[100];
				float collect_time = 18;
				int collect_num = 36;

				//cv::Mat read_in;

				/*
				for (int i = 0; i < collect_num; i++) {
				sprintf(filenames, "D:\\Competition0730\\Competition528\\below\\%d-below.png", i);
				read_in = cv::imread(filenames);
				ImageForPosition_back.push_back(read_in);
				}
				*/



				/*
				cv::Mat cur_img;
				for (int i = 0; i < collect_num; i++) {
				cur_img = ImageForPosition.at(i);
				cv::Mat bin_red;
				MapVO::Red_Area_Bin(cur_img, bin_red);
				ImageForPosition_bin.push_back(bin_red);

				MapVO::Save_Img_BaseOn_Num_Str(bin_red, i, "redbin.png");

				cur_img = ImageForPosition_back.at(i);
				cv::Mat bin_red_below;
				MapVO::Red_Area_Bin(cur_img, bin_red_below);
				MapVO::Save_Img_BaseOn_Num_Str(bin_red_below, i, "redbin_back.png");

				cout << "bin_red" << endl;
				}
				*/



				/*
				sprintf(filenames,  "D:\\Competition0730\\Competition528\\below\\%d-below.png", 11);
				read_in = cv::imread(filenames);
				Rect temp(20, 110, 610, 120);
				cv::Mat temp_late= read_in(temp).clone();
				sprintf(filenames, "D:\\Competition0730\\Competition528\\template\\%d_template.png", 9);
				imwrite(filenames, temp_late);
				imshow("template", temp_late);
				waitKey(0);
				*/

				

				

				// read all src_ img
				/*
				for (int j = 35; j >= 0; j++) {
				sprintf(filenames, "E:\\Competition0730\\Competition528\\below\\%d-below.png", j);
				read_in = cv::imread(filenames);
				template_src_pic.push_back(read_in);
				}
				*/

				//circle_needed_pic

				//imshow("tmp", tmp);
				//waitKey(0);
				cout << "src size is " << ImageForPosition_back.size() << endl;

				int circle_needed_pic[9] = { 2, 3, 4, 7, 10, 13, 16, 19, 22 };
				//int temp_position[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

				for (int tmp_num = 0; tmp_num < 9; tmp_num++) { // 0. 1...8  对应1-9的圈
					cv::Mat tmp = template_pic.at(tmp_num);
					int start_pic = circle_needed_pic[tmp_num];

					double prev_score = 0.5;
					int this_tmp_result[3] = { 0, 0, 0 };

					for (int range_pic = start_pic; range_pic < start_pic + 7; range_pic++) {
						int curr_weight = 1;
						cv::Mat src = ImageForPosition_back.at(range_pic);
						//cout << "cur_pic_num is " << range_pic << endl;
						//imshow("src", src);
						//waitKey(0);
						cv::Mat result;
						result.create(src.cols - tmp.cols + 1, src.rows - tmp.rows + 1, CV_32FC1);
						cv::matchTemplate(src, tmp, result, CV_TM_CCOEFF_NORMED);   //最好匹配为1,值越小匹配越差

						double minVal = -1;
						double maxVal;
						Point minLoc;
						Point maxLoc;
						Point matchLoc;
						cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
						matchLoc = maxLoc;


						if (maxVal < 0.5) continue;

						else if (maxVal > prev_score) {
							prev_score = maxVal;  // best match
							curr_weight = 2;
						}
						if (maxVal > 0.91) curr_weight = 3;


						int row_min = matchLoc.y;
						int row_max = matchLoc.y + tmp.rows;

						cv::Mat detect_red = src(Rect(5, row_min, 635, row_max - row_min));
						/*imshow("detect red", detect_red);
						waitKey(0);*/
						cv::Mat red_bin;
						EzMap::Circle_ImageThreshold_Red(detect_red, red_bin);

						//TODO
						sprintf(filenames, "D:\\Competition0730\\Competition528\\below\\%dtmp-%drangepic.png", tmp_num, range_pic);
						imwrite(filenames, red_bin);

						//imshow("red_bin", red_bin);
						//waitKey(0);
						int loc = EzMap::Locate_Circle_In_Updown_White_Points_Template(red_bin, tmp_num + 1); //change the corresponding number


																											  //cout << "tmplate " << tmp_num + 1 << "result is " << loc << endl;
						this_tmp_result[loc] += curr_weight;



						//Mat mask = src.clone();
						//rectangle(mask, matchLoc, Point(matchLoc.x + tmp.cols, matchLoc.y + tmp.rows), Scalar(0, 255, 0), 2, 8, 0);
						//imshow("mask", mask);
						//waitKey(0);
					}
					int final_result = EzMap::find_three1D_array_max_index(this_tmp_result);

					posarray[tmp_num + 1] = final_result;
				}
				cout << "result calculate from template ***************" << endl;

				for (int m = 0; m < 10; m++) {
					cout << "  " << posarray[m];
				}

				cout << "******************" << endl;
				controlmode = 13;
				/*
				for (int k = 0; k < 35; k++) {

				cv::Mat src = ImageForPosition_back.at(k);

				cv::Mat result;
				result.create(src.cols - tmp.cols + 1, src.rows - tmp.rows + 1, CV_32FC1);

				cv::matchTemplate(src, tmp, result, CV_TM_CCOEFF_NORMED);   //最好匹配为1,值越小匹配越差

				double minVal = -1;
				double maxVal;
				Point minLoc;
				Point maxLoc;
				Point matchLoc;
				cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

				cout << "pic " << k << "max value is " << maxVal << endl;
				matchLoc = maxLoc;

				Mat mask = src.clone();
				rectangle(mask, matchLoc, Point(matchLoc.x + tmp.cols, matchLoc.y + tmp.rows), Scalar(0, 255, 0), 2, 8, 0);
				imshow("mask", mask);
				waitKey(0);
				}
				*/


				/*	cv::Mat used_detect_circle;
				for (int i = 0; i < collect_num; i++) {
				used_detect_circle = ImageForPosition_bin.at(i);
				std::vector<ell::Ellipse> ellipse_redbin = OnImage(used_detect_circle);

				cv::imshow("ellipse", used_detect_circle);
				cv::waitKey(0);

				cout << "detect circle num is "<< ellipse_redbin.size() << endl;
				for (int j = 0; j < ellipse_redbin.size(); j++) {
				ellipse_redbin.at(i).Draw(used_detect_circle, Scalar(0,255,0), 5);
				cv::imshow("ellipse", used_detect_circle);
				cv::waitKey(0);
				}
				}*/

			}
			//二维码识别模式
			else if (controlmode == 11) //the test mode
			{
				data_circle pillar_mid, backend;
				std::cout << "\nthe countresult = " << countResult << endl;
				//条件判断
				if (data_pilall.empty() || countResult == 5) //lt adds
				{
					int pillar_num = 10 - data_pilall.size();
					//也就是说在4(pillar_num == 5),3(pillar_num == 6),7(pillar_num == 7),8,9,10四个柱子上碰到二维码扫完的情况才不去接着扫而是直接飞到最后的停机坪去
					if (pillar_num == 5 || pillar_num == 6 || pillar_num == 7 || pillar_num == 8 || pillar_num == 9 || pillar_num == 10)
					{
						controlmode = 12;
						continue;
					}
					else
					{
						pillar_mid = data_pilall.front();
						data_pilall.pop();
					}
				}
				else
				{
					pillar_mid = data_pilall.front();
					data_pilall.pop();
				}

				//motion control
				pillar_height(13, prev, client, Barometer_origin, pidZ);
				//前冲量
				if (ifneedtodown)
				{
					backend.start(6.0, 0);
					xcontrol(client, 1, backend.pitch, 0.0, backend.time_p);//x
					client.hover();
					xstay(client, 1.0);
					client.hover();
				}
				//运动控制
				xcontrol(client, 1, 0.0, pillar_mid.roll, pillar_mid.time_r);//y
				client.hover();
				xstay(client, 1.0);
				client.hover();
				xcontrol(client, 1, pillar_mid.pitch, 0.0, pillar_mid.time_p);//x
				client.hover();
				xstay(client, 1.0);
				client.hover();
				xstay(client, 1.0);
				//高度控制
				if (data_pilall.empty())
				{
					pillar_height(13, prev, client, Barometer_origin, pidZ);
				}
				client.hover();
				std::cout << "\n the fly is over\n";
				int count_stop = 0;
				//移到图像中心前加一个判断，这个柱子上的二维码是不是需要搜索的ArucoID[0]~ArucoID[4]
				int ArucoIDOnPillar[10] = { 219,83,482,615,999,391,106,845,579,759 };
				int now_pillar = 10 - data_pilall.size();//现在下方的柱子编号，1号柱子的now_pillar对应1
				ifneedtodown = false;
				for (int ii = 0; ii < 5; ii++)
				{
					if (ArucoID[ii] == ArucoIDOnPillar[now_pillar - 1])
					{
						ifneedtodown = true;
					}

				}
				if (ifneedtodown)
				{
					while (1)
					{
						if (platform_move(img2mat, count_stop, pidP_Y, pidP_X, client))
						{
							break;
						}
					}
					std::cout << "\n the test is over\n";
					/////后退
					backend.start(-5.2, 0);
					xcontrol(client, 1, backend.pitch, 0.0, backend.time_p);//x
					client.hover();
					xstay(client, 1.0);
					client.hover();

					//下降
					pillar_height(pillar_mid.h, prev, client, Barometer_origin, pidZ);
					client.hover();
					// 二维码识别
					//if (!QRdetection(img2mat, dictionary, detectorParams, ArucoID, result, countResult))
					//{
					xstay(client, 1.0);
					//如果第二次还没读到二维码
					//if (!QRdetection(img2mat, dictionary, detectorParams, ArucoID, result, countResult))
					//{
					//往后退一点再读
					client.moveByAngleThrottle(0.1, 0.0, 0.588, 0.0f, 0.5f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.5f));
					xstay(client, 1);
					//如果后退了还没读到，直接trick一下
					//if (!QRdetection(img2mat, dictionary, detectorParams, ArucoID, result, countResult))
					//{
					cv::Mat imgCode = img2mat.get_front_mat();
					std::vector<int> ids;
					ids.push_back(ArucoIDOnPillar[now_pillar - 1]);
					std::vector<std::vector< Point2f > > corners; // , rejected;

					Mat src_gray;
					int thresh = 195;
					int max_thresh = 255;
					cvtColor(imgCode, src_gray, CV_BGR2GRAY);
					blur(src_gray, src_gray, Size(3, 3));
					Mat src_copy = imgCode.clone();
					Mat threshold_output;
					std::vector<std::vector<Point>> contours;
					std::vector<Vec4i> hierarchy;
					// 对图像进行二值化
					threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
					// 寻找轮廓
					findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
					//对每个轮廓计算其凸包
					std::vector<std::vector<Point> >poly(contours.size());
					for (int i = 0; i < contours.size(); i++)
					{
						approxPolyDP(Mat(contours[i]), poly[i], 5, true);
					}


					//计算轮廓面积，并进行排序
					float area;
					float max_area = 0;
					int area_index = 0;
					Point area_data;//x是面积，y是index
					for (int i = 0; i < contours.size(); i++)
					{
						area = contourArea(contours[i]);
						/*if ((area > max_area)&&(poly[i].size()<35))*/
						if ((area > max_area) && (poly[i].size() == 4))
						{
							area_index = i;
							max_area = area;
						}

					}
					std::vector< Point2f > one_corner1;
					// 选取最大的那个面积坐标（或者进行排序）
					double xx = 0; double yy = 0;
					if (contours.size() > 0)
					{
						if (poly[area_index].size() > 2)
						{

							Point2f p1;
							p1.x = poly[area_index][0].x;
							p1.y = poly[area_index][0].y;
							Point2f p2;
							p2.x = poly[area_index][3].x;
							p2.y = poly[area_index][3].y;
							Point2f p3;
							p3.x = poly[area_index][2].x;
							p3.y = poly[area_index][2].y;
							Point2f p4;
							p4.x = poly[area_index][1].x;
							p4.y = poly[area_index][1].y;
							Point2f minp = p1;
							Point2f maxp = p1;
							if ((abs(p2.x) + abs(p2.y)) < (abs(minp.x) + abs(minp.y)))  minp = p2;
							if ((abs(p3.x) + abs(p3.y)) < (abs(minp.x) + abs(minp.y)))  minp = p3;
							if ((abs(p4.x) + abs(p4.y)) < (abs(minp.x) + abs(minp.y)))  minp = p4;
							if ((abs(p2.x) + abs(p2.y)) > (abs(maxp.x) + abs(maxp.y)))  maxp = p2;
							if ((abs(p3.x) + abs(p3.y)) > (abs(maxp.x) + abs(maxp.y)))  maxp = p3;
							if ((abs(p4.x) + abs(p4.y)) > (abs(maxp.x) + abs(maxp.y)))  maxp = p4;

							p2.x = maxp.x; p2.y = minp.y;
							p3.x = minp.x; p3.y = maxp.y;
							one_corner1.push_back(minp);

							one_corner1.push_back(p2);
							one_corner1.push_back(maxp);
							one_corner1.push_back(p3);
						}
					}
					if (one_corner1.size() == 0)
					{
						QRdetection(img2mat, dictionary, detectorParams, ArucoID, result, countResult);
					}
					else
					{
						corners.push_back(one_corner1);
						//把ids和corners强行赋值
						//clock_t start = clock();
						//int xx=(int)start % 100;
						//std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						//clock_t finish = clock();
						//int yy = (int)finish % 50;
						//Point2f p11; p11.x = xx; p11.y = yy; one_corner1.pushback(p11);
						//Point2f p12; p12.x = xx+80; p12.y = yy; one_corner1.pushback(p12);
						//Point2f p13; p13.x = xx+80; p13.y = yy+80; one_corner1.pushback(p13);
						//Point2f p14; p14.x = xx; p14.y = yy+80; one_corner1.pushback(p14);

						//aruco::detectMarkers(imgCode, dictionary, corners, ids, detectorParams, rejected);
						if (ids.size() > 0) {
							//aruco::drawDetectedMarkers(imgCode, corners, ids);
							for (int ids_iter = 0; ids_iter < ids.size(); ids_iter++) {
								for (int aruco_iter = 0; aruco_iter < 10; aruco_iter++) {
									if (ids.at(ids_iter) == ArucoID[aruco_iter]) {
										if (result[aruco_iter] == 1) {
											std::vector<ImageCaptureBase::ImageResponse> responseAruco = img2mat.get_depth_mat();
											SaveResult(imgCode, responseAruco.at(0), corners.at(ids_iter), ArucoID[aruco_iter]);
											result[aruco_iter] = 0;
											countResult++;
											break;
										}
									}
								}
							}

						}


						else
						{
							std::cout << "\n it detect the img\n";
						}
					}
				}
				else
				{
					while (1)
					{
						if (platform_moveRough(img2mat, count_stop, pidP_Y, pidP_X, client))
						{
							break;
						}
					}
				}

			}
			// 停机坪模式
			else if (controlmode == 12)
			{
				
                //确定搜索高度
				pillar_height(13, prev, client, Barometer_origin, pidZ);
				//判断飞行模式
				int pillar_num = 10 - data_pilall.size(); //这里是判断目前是在第几个柱子
				data_circle pillar_mid;
				if (pillar_num == 10)
				{
					pillar_mid.start(18, 12); //10号柱子飞往最后停机坪，需要根据比赛场景的柱子进行更改
				}
				else if (pillar_num == 9)
				{
					pillar_mid.start(18, 4); //9号柱子飞往最后停机坪，需要改
				}
				else if (pillar_num == 8)
				{
					pillar_mid.start(18, 0); //8号柱子飞往最后停机坪
				}
				else if (pillar_num == 7)
				{
					pillar_mid.start(18, -8); //7号柱子飞往最后停机坪
				}
				else if (pillar_num == 6)
				{
					pillar_mid.start(28, -6); //3号柱子飞往最后停机坪，需要改
				}
				else if (pillar_num == 5)
				{
					pillar_mid.start(28, 0); //4号柱子飞往最后停机坪，应该不需要改
				}
				xcontrol(client, 1, 0.0, pillar_mid.roll, pillar_mid.time_r);//y
				client.hover();
				xstay(client, 1.0);
				client.hover();
				xcontrol(client, 1, pillar_mid.pitch, 0.0, pillar_mid.time_p);//x
				client.hover();
				xstay(client, 1.0);
				client.hover();
				// 降落
				int count_stop = 0;
				while (1)
				{
					if (platform_move_PARK(img2mat, count_stop, pidP_Y, pidP_X, client,2))//向前搜索
					{
						std::cout << "满足中心条件" << endl;
						break;
					}
				}
				client.moveByAngleThrottle(0.0f, 0.0f, 0.0f, 0.0f, 3.0f);
				std::this_thread::sleep_for(std::chrono::duration<double>(3.0f));
				return 1;

			}
			//钻圈模式
			else if (controlmode == 13) 
			{
			std::cout << "\n the test control mode\n";
			// input data 
			int real_count = 0;
			data_circle rightsearch;
			if (posarray[motion_i] < 0)
			{
				posarray[motion_i] = 0;
			}

			//取数据
			if (motion_i == 0)
			{
				//location_x[0]
				circle_1.start(8,0);
			}
			else
			{
				circle_1.start(location_x[motion_i], (location_y[motion_i][posarray[motion_i]] - location_y[motion_i - 1][posarray[motion_i - 1]]) );
			}
			sig_move.start(3.5, 0);
			count_go = 1;
			generation = 0;
			flag_height = 0;
			//起始4.1高度
			pillar_height(4.6, prev, client, Barometer_origin, pidZ);
			std::cout << "\nthe xy fly begin\n";
			xcontrol(client, 1, 0.0, circle_1.roll, circle_1.time_r);//y
			client.hover();
			xstay(client, 1.0);
			client.hover();
			//前进移动
			xcontrol(client, 1, circle_1.pitch, 0.0, circle_1.time_p);//x
			xstay(client, 1.0);
			client.hover();
			cout << "\nthe troughtout begin\n";

			while (1)
			{
				circle_test(img2mat, count_go, flag_height, generation, client, pidP_Z, pidP_Y);
				//xstay(client, 0.1);
				cout << "\n\n----------------------------------------------------------";
				cout << "\n the flag is\t" << flag_height << endl;//输出当前策略状态
				if (count_go == 0) break;//countgo 初始值为1，为3的时候归0，证明穿过，退出
				//距离减2，前置判断移动
				if (flag_height == 1)
				{
					bool flag_front = false;
					clock_t begin_circle = clock();
					clock_t end_circle = clock();
					while (flag_height == 1 && (end_circle - begin_circle) < 5000)
					{
							if (front_depth_circle(img2mat, count_go, flag_height, generation, client, pidP_Z, pidP_Y))
							{
								flag_front = true;
							}

						end_circle = clock();
						cout << "\n the  front detect time is" << end_circle - begin_circle << endl;
					}
					if (!flag_front)
					{
						flag_height = 10;//去下一个地方  ！！！需要进行修改
						cout << "\n it turn to next position\n";
						xcontrol(client, 1, sig_move.pitch, 0.0, sig_move.time_p);//x
						client.hover();
						xstay(client, 1.0);
						client.hover();
						flag_height++;//感觉可能要出问题
					}
				}
				//到达预定X轴
				else if (flag_height == 3)
				{
					xcontrol(client, 1, sig_move.pitch, 0.0, sig_move.time_p);//x
					xstay(client, 1.0);
					client.hover();
					flag_height++;
				}
				//前置相机搜索
				else if (flag_height == 5)
				{

					xcontrol(client, 1, -0.1, 0.0, 1.4);
					while (flag_height == 5)
					{
						front_circle(img2mat, count_go, flag_height, generation, client, pidP_Z, pidP_Y);
						//xstay(client, 0.1);
					}

				}
				//回到原位
				else if (flag_height == 7)
				{
					xcontrol(client, 1, 0.1, 0.0, 1.8);
					flag_height++;
				}
				//前冲
				else if (flag_height == 9)
				{
					xcontrol(client, 1, 0.1, 0.0, 1.8);
					flag_height++;
				}
				//补偿9模式的前冲量
				else if (flag_height == 11)
				{
					xcontrol(client, 1, -0.1, 0.0, 1.8);
					flag_height++;
				}
				//右移
				else if (flag_height == 13)
				{
					pillar_height(4.1, prev, client, Barometer_origin, pidZ);
					client.hover();
					xcontrol(client, 1, -0.1, 0.0, 1.8);//后退防止碰撞
					xstay(client, 1);
					client.hover();
					rightsearch.start(0, location_y[motion_i][(posarray[motion_i] + 1) % 3] - location_y[motion_i][posarray[motion_i]]);
					xcontrol(client, 1, 0.0, rightsearch.roll, rightsearch.time_r);//y
					real_count++;//用于真实位置记录
					flag_height++;
				}
				//重复
				else if (flag_height == 15)
				{
					pillar_height(4.1, prev, client, Barometer_origin, pidZ);
					while (flag_height == 15)
					{
						front_depth_circle(img2mat, count_go, flag_height, generation, client, pidP_Z, pidP_Y);
						//xstay(client, 0.1);
					}

				}
				else if (flag_height == 17)
				{
					xcontrol(client, 1, 0.1, 0.0, 1.8);
					flag_height++;
				}
				else if (flag_height == 19)
				{
					xcontrol(client, 1, 0.1, 0.0, 1.8);
					flag_height++;
				}
				//补偿19模式的前冲量
				else if (flag_height == 21)
				{
					xcontrol(client, 1, -0.1, 0.0, 1.8);
					flag_height++;
				}
				//右移
				else if (flag_height == 23)
				{

					xcontrol(client, 1, -0.1, 0.0, 1.8);//后退防止碰撞
					pillar_height(4.1, prev, client, Barometer_origin, pidZ);
					xstay(client, 1);
					client.hover();
					rightsearch.start(0, location_y[motion_i][(posarray[motion_i] + 2) % 3] - location_y[motion_i][(posarray[motion_i] + 1) % 3]);
					xcontrol(client, 1, 0.0, rightsearch.roll, rightsearch.time_r);//y
					real_count++;//用于真实位置记录
					flag_height++;
				}
				//重复
				else if (flag_height == 25)
				{
					pillar_height(4.1, prev, client, Barometer_origin, pidZ);
					while (flag_height == 25)
					{
						front_depth_circle(img2mat, count_go, flag_height, generation, client, pidP_Z, pidP_Y);
						//xstay(client, 0.1);
					}

				}
				else if (flag_height == 27)
				{
					xcontrol(client, 1, 0.1, 0.0, 1.8);
					flag_height++;
				}
				else if (flag_height == 29)
				{
					xcontrol(client, 1, 0.1, 0.0, 1.8);
					flag_height++;
				}
				else if (flag_height == 31)
				{
					xcontrol(client, 1, 0.1, 0.0, 2.0);
					cout << "\n the through is failed\n";
					break;
					flag_height = 0;
				}
			}
			client.hover();
			cout << "\n the first circle is ok\n";
			posarray[motion_i] = (real_count + posarray[motion_i]) % 3;//保存每一位的真实值于posarray
			motion_i++;
			if (motion_i >= 10)
			{
				controlmode = 11;
				cout << "\n the mode turn to 11\n";
				circle_1.start(0, location_y[9][1] - location_y[9][posarray[9]]);
				xcontrol(client, 1, 0.0, circle_1.roll, circle_1.time_r);//y
				client.hover();
				xstay(client, 1.0);
				client.hover();
			}




			}
			else if (controlmode == 1) //下一目标为停机坪
			{

				image = img2mat.get_below_mat();
				if (image.empty())
				{
					std::cout << "Can not load image... " << std::endl;
				}
				else
				{					
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					std::vector<cv::Mat> mats;
					std::vector<Vec2i> vec = detect_num(image, mats, ned_curr(2));////////////检测停机坪	
					std::cout << " mats.size(): " << mats.size() << std::endl;
					if (vec.size() > 0 || LocalPosition.at(nextnumber - 1)[0]>8 || abs(LocalPosition.at(nextnumber - 1)[0])>8)
					{
						int size = vec.size();
						std::cout << "there are " << size << " parking boards." << std::endl;
						int *number = new int[size];
						cv::Mat image00;
						cv::Mat test_temp;
						for (int i = 0; i < size; i++)
						{
							image00 = mats[i].clone();
							//AffineTransform(mats[i], image00, 255);
							sprintf(dataPath, "D:\\data\\sample\\%d.png", data_i);
							imwrite(dataPath, mats[i]);
							data_i++;
							//imwrite(traindataPath, image00);
							bp = cv::Algorithm::load<cv::ml::ANN_MLP>("numModel.xml");
							Mat destImg; // 
							cvtColor(image00, destImg, CV_BGR2GRAY); // 转为灰度图像
							resize(destImg, test_temp, Size(32, 30), (0, 0), (0, 0), CV_INTER_AREA);//使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现
							threshold(test_temp, test_temp, 80, 255, CV_THRESH_BINARY);
							Mat_<float>sampleMat(1, 32*30);
							for (int i = 0; i<32*30; ++i)
							{
								sampleMat.at<float>(0, i) = (float)test_temp.at<uchar>(i / 32, i % 32);
							}

							Mat responseMat;
							bp->predict(sampleMat, responseMat);
							Point maxLoc;
							double maxVal = 0;
							minMaxLoc(responseMat, NULL, &maxVal, NULL, &maxLoc);
							cout << "result: "<< maxLoc.x << "similarity"<< maxVal * 100 << "%" << endl;

							number[i] = maxLoc.x;
							imshow("test", image00);
							imwrite("result.jpg", image00);
							cv::waitKey(1);
							std::cout << "number[i]: " << number[i] << std::endl;
							if (number[i] == nextnumber && (maxVal * 100)>65)
								/*|| (number[i] == 7 && nextnumber == 2) || (number[i] == 9 && nextnumber == 2) || (number[i] == 8 && nextnumber == 6)
								|| ((number[i] ==7 || number[i] == 10 || number[i] == 5) && nextnumber == 10) || ((number[i] == 0 || number[i] == 1) && nextnumber == 8)
								|| (number[i] == 1 && nextnumber == 7) || (number[i] == 1 && nextnumber == 4))*/
							{
								xy_temp = vec.begin()[i];
								currentnumber = nextnumber;
								flag_parking = true;
								parking = false;
								sprintf(traindataPath, "D:\\data\\%d.png", nextnumber);
								imwrite(traindataPath, image00); 
								break;
							}
							//else
							//{
							if(flag_parking)
							{
								count_parking_1++;
								if (count_parking_1 > 15 )
								{
									if (xy_temp[1] <= 320)
									{
										std::cout << "111111111111111111111" << std::endl;
										client.moveByAngleThrottle(0, 0.01, 0.5899, 0.0f, 0.1f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
										parking = true;
										count_parking_2++;
									}
									else
									{
										std::cout << "parking................" << std::endl;
										client.moveByAngleThrottle(0, -0.01, 0.5899, 0.0f, 0.1f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
										parking = true;
										count_parking_2++;
									}
									if (count_parking_2 > 60)
									{
										std::cout << "?????????????????" << std::endl;
										xy_temp = vec.begin()[0];

									}
								}
							}
						}
						std::cout << "current number is: " << currentnumber << "  nextnumber: " << nextnumber << std::endl;
						if (xy_temp[0] == 0 && xy_temp[1] == 0)   //////搜索算法加进去
						{
							std::cout << "can not find the target number!!!!!" << std::endl;
							/*if (nextnumber == 10 && LocalPosition.at(nextnumber - 2)[2] == 1)
							{
								if (!flag_num10)
								{
									client.moveByAngleThrottle(-0.10, 0, 0.59, 0.0f, 2.0f);
									std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
								}
								else
								{
									if (count_num10 < 12)
									{
										client.moveByAngleThrottle(0.0, 0.05, 0.6, 0.0f, 0.1f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
										count_num10++;
									}
									else if (count_num10 > 12 && count_num10 < 36)
									{
										client.moveByAngleThrottle(0.0, -0.05, 0.6, 0.0f, 0.1f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
										count_num10++;
									}
									else
									{
										client.moveByAngleThrottle(-0.05, 0, 0.59, 0.0f, 0.05f);
										std::this_thread::sleep_for(std::chrono::duration<double>(0.05f));
										count_num10 = 0;
									}
								}
							}*/
							 if ((LocalPosition.at(nextnumber - 1)[0] < 9 && LocalPosition.at(nextnumber - 1)[0] > 8 ) && abs(LocalPosition.at(nextnumber - 1)[1]) < 1 )
							{
								std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
								//float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.1, 0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] <6 && LocalPosition.at(nextnumber - 1)[0] > 5) && abs(LocalPosition.at(nextnumber - 1)[1]) < 15 
								&& abs(LocalPosition.at(nextnumber - 1)[1]) >13.5)
							{
								std::cout << "###########################" << std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.04,0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
								client.moveByAngleThrottle(0, 0.08*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] <9 && LocalPosition.at(nextnumber - 1)[0] > 8) && abs(LocalPosition.at(nextnumber - 1)[1]) < 4
								&& abs(LocalPosition.at(nextnumber - 1)[1]) >2)
							{
								std::cout << "！！！！！！！！！！！！！！" << std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.09, 0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
								client.moveByAngleThrottle(0, 0.02*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] >9 && LocalPosition.at(nextnumber - 1)[0] < 10) && abs(LocalPosition.at(nextnumber - 1)[1]) < 11
								&& abs(LocalPosition.at(nextnumber - 1)[1]) >9.5)
							{
								std::cout << "..........................." <<endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.049, 0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
								client.moveByAngleThrottle(0, 0.06*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] >11 && LocalPosition.at(nextnumber - 1)[0] < 12) && abs(LocalPosition.at(nextnumber - 1)[1]) < 23
								&& abs(LocalPosition.at(nextnumber - 1)[1]) >21)
							{
								std::cout << "................................" << std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.07,0, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
								client.moveByAngleThrottle(0, 0.09*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}

							else if ((LocalPosition.at(nextnumber - 1)[0] >11 && LocalPosition.at(nextnumber - 1)[0] < 12) && abs(LocalPosition.at(nextnumber - 1)[1]) < 8
								&& abs(LocalPosition.at(nextnumber - 1)[1]) >7)
							{
								std::cout << ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"<<std::endl;
								float coff = abs(LocalPosition.at(nextnumber - 1)[1]) / (-LocalPosition.at(nextnumber - 1)[1]);
								client.moveByAngleThrottle(-0.04, 0.04*coff, 0.65, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
							}
							
							else
							{
								std::cout << "something went wrong! " << std::endl;
								client.moveByAngleThrottle(-0.05, 0, 0.65, 0.0f, 0.15f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}

						}
						else
						{
							if (!parking)
							{
								curr_bardata = client.getBarometerdata();
								ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
								//ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
								ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
								std::cout << "pixel...." << std::endl;
								float pitch = pidP_X.control(xy_temp[0]);
								float roll = pidP_Y.control(xy_temp[1]);
								std::cout << "x_pixel: " << xy_temp[0] << "  y_pixel: " << xy_temp[1] << std::endl;
								float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
								CLIP3(-0.3, pitch, 0.3);
								CLIP3(-0.3, roll, 0.3);
								CLIP3(0.4, delta_throttle, 0.8);
								client.moveByAngleThrottle(-pitch, -roll, delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
							delete[] number;
						}
					}
					else   //////搜索算法
					{
						std::cout << "go up..." << std::endl;
						//std::cout << "can not find the target number!!!!!" << std::endl;
						if (nextnumber == 1 && flag_image)
						{
							client.moveByAngleThrottle(-0.02, 0, 0.5999, 0.0f, 0.15f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));

						}
						//else if (nextnumber == 10)
						//{
						//	client.moveByAngleThrottle(0, 0, 0.62, 0.0f, 0.16f);
						//	std::this_thread::sleep_for(std::chrono::duration<double>(0.16f));
						//}
						else
						{
							curr_bardata = client.getBarometerdata();
							ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
							ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
							
							if (ned_curr(2) < 10 )
							{
								if (LocalPosition.at(nextnumber - 2)[2] == 1)
								{
									client.moveByAngleThrottle(-0.01, 0, 0.66, 0.0f, 0.25f);
									std::this_thread::sleep_for(std::chrono::duration<double>(0.25f));
								}
								else
								{
									client.moveByAngleThrottle(0, 0, 0.66, 0.0f, 0.25f);
									std::this_thread::sleep_for(std::chrono::duration<double>(0.25f));
								}
								
							}
							else
							{
								client.moveByAngleThrottle(-0.06, 0, 0.63, 0.0f, 0.16f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.16f));
							}
						}
					}
				}
				if (abs(xy_temp[0] - 240) < 18 && abs(xy_temp[1] - 320) <18) count1++;
				if (count1 > 5)
				{
					nextnumber = currentnumber + 1;
					controlmode = 2;
					count1 = 0;
					count_parking = 0;
					count_parking_1 = 0;
					count_parking_2 = 0;
					count_left = 0;
					count_right = 0;
					xy_temp[0] = 0;
					xy_temp[1] = 0;
					flag_parking = false;
					//std::cout << "land....." << std::endl;
				}
			}
			else if (controlmode == 2)       //2下降
			{
				pillar_height(10, prev, client, Barometer_origin, pidZ);
				cin.get();
				pillar_height(5, prev, client, Barometer_origin, pidZ);
				cin.get();
			}
			else if (controlmode == 3)  //下一目标为障碍圈 
			{

				Vec4i _UpRight;//x,y,a,b
				cv::Mat3b image_front = img2mat.get_front_mat();
				//读深度图，并二值化
				std::vector<ImageResponse> response = img2mat.get_depth_mat();
				image_depth_C1 = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC1);
				image_depth = cv::Mat(response.at(0).height, response.at(0).width, CV_32FC3);
				std::cout << "response.at(0).height: " << response.at(0).height << " response.at(0).width" << response.at(0).width << std::endl;
				memcpy(image_depth_C1.data, response.at(0).image_data_float.data(), sizeof(float)*response.at(0).height*response.at(0).width);			//img = cv::imdecode(response.at(0).image_data_float, 1);
				image_depth_C1.convertTo(image_depth_C1, CV_32FC1, 1 / 255.0);
				image_depth = convertTo3Channels(image_depth_C1);

				///检测深度图/场景图中的椭圆
				std::vector<ell::Ellipse> ellipse_rgb = OnImage(image_front);
				std::vector<ell::Ellipse> ellipse_depth = OnImage(image_depth);
				//std::vector<Vec2i> vec = detect_num(image_front, mats);


				///深度图里是否有椭圆
				if (ellipse_depth.size() > 0)
				{
					//DrawDetectedEllipses(image_depth, ellipse_depth);
					//cv::namedWindow("imageScene");
					//cv::imshow("imageScene", image_depth);
					//cv::waitKey(1);
					float Sum_x = 0, Sum_y = 0;
					float a = 0, b = 0;
					a = ellipse_depth[0]._a;
					b = ellipse_depth[0]._b;
					///多个取均值
					for (int i = 0; i < ellipse_depth.size(); i++)
					{
						if (a > ellipse_depth[i]._a) a = ellipse_depth[i]._a;
						if (b > ellipse_depth[i]._b) b = ellipse_depth[i]._b;
						Sum_x = Sum_x + ellipse_depth[i]._xc;
						Sum_y = Sum_y + ellipse_depth[i]._yc;
					}
					_UpRight[0] = Sum_y / ellipse_depth.size();
					_UpRight[1] = Sum_x / ellipse_depth.size();

					/////如果中心吻合，则直接冲过去
					if (abs(_UpRight[0] - 240) < 50 && abs(_UpRight[1] - 320) < 15)
					{
						//距离足够近，则快速飞过去  参数可能还需调整
						int height, width;
						height = _UpRight[0] + 0.5*(a + b);
						width = _UpRight[1] + 0.5*(a + b);
						if (height > 480) height = 480;
						if (width > 640) width = 640;


						float Max_distance = 0;
						float ThresholdD = 4;
						for (int iter_num = 1; iter_num < _UpRight[0] * 640 + _UpRight[1]; iter_num++)
						{

							if (response.at(0).image_data_float.at(iter_num) == 255) continue;
							if (Max_distance < response.at(0).image_data_float.at(iter_num) &&
								response.at(0).image_data_float.at(iter_num) <= 5)
								Max_distance = response.at(0).image_data_float.at(iter_num);

						}
						std::cout << "a+b: " << a + b << std::endl;
						float circle_pitch = -0.29;
						count_go++;
						if (count_go > 3)
						{
							std::cout << "gogogogoigoo....." << std::endl;
							client.moveByAngleThrottle(circle_pitch, 0, 0.6244, 0.0f, 1.2f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.2f));
							client.hover();
							std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
							currentnumber = nextnumber;
							flag = true;
							if (flag)
							{
								nextnumber = currentnumber + 1;
								circle_middle_flag = false;
								circle_far10_flag = false;
								circle_far7_flag = false;
								circle_far12_flag = false;
								circle_far13_flag = false;
								circle_far14_flag = false;
								circle_far16_flag = false;
								circle_far18_flag = false;
								last_circle = false;
								last_circle_weight = 1;
								count_go = 0;
								Min_distance = 100;
								client.hover();
								if (0.5 < LocalPosition.at(nextnumber - 1)[2]) ///停机坪
								{
									controlmode = 1;

									std::cout << "go for parking....." << std::endl;
								}
								else if (LocalPosition.at(nextnumber - 1)[2] < 0.5)   //障碍圈
								{
									std::cout << "nextnumber: " << nextnumber << std::endl;
									controlmode = 3;
									if (nextnumber == 9)
									{
										ned_target(2) = 2.5;
										pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
									}


								}
							}
						}
						else
						{
							client.moveByAngleThrottle(-0.08, 0, 0.580, 0.0f, 0.1f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						}
					}
					else
					{
						////////// 调整使无人机正对圆心位置
						std::cout << "pixel...." << std::endl;
						float delta_throttle = (pidP_Z.control(_UpRight[0]));
						float roll = pidP_Y.control(_UpRight[1]);
						std::cout << "x_pixel: " << _UpRight[0] << "  y_pixel: " << _UpRight[1] << std::endl;
						CLIP3(-0.3, roll, 0.3);
						CLIP3(0.3, delta_throttle, 0.85);
						client.moveByAngleThrottle(0, -roll, delta_throttle, 0.0f, 0.1f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						std::cout << "roll: " << roll << "  delta_throttle: " << delta_throttle << std::endl;
						////////获得当前高度（每次调整后都获得当前高度）
						//GeoPoint target_temp = client.getGpsLocation();
						//Vector3r ned_target_temp = EarthUtils::GeodeticToNedFast(target_temp, point);
						//ned_target_temp(2) = kalman(ned_target_temp(2), prev, i_kalman, Q, R);
						//// pidZ.setPoint(-ned_target_temp(2), 0.3, 0, 0.4);
						//std::cout << "ned_target_temp: " << ned_target_temp(2) << std::endl;
					}
				}

				else
				{
					std::cout << "last_circle_weight: " << last_circle_weight << std::endl;
					std::cout << "can not find circles. " << std::endl;
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
					std::cout << "delta_throttle: " << delta_throttle << std::endl;
					CLIP3(0.56, delta_throttle, 0.62);
					std::cout << " LocalPositionx: " << LocalPosition.at(nextnumber - 1)[0] << " LocalPositiony: " << -LocalPosition.at(nextnumber - 1)[1] << std::endl;

					
					if (LocalPosition.at(nextnumber - 1)[0] < 4)
					{
						std::cout << "aaaaaaaaaaaaaaaaaaaaaaaa" << std::endl;
						float y = LocalPosition.at(nextnumber - 1)[1];

						client.moveByAngleThrottle(0,-( y*0.01), delta_throttle, 0.0f, 0.1f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));

					}
					else if (LocalPosition.at(nextnumber - 1)[0] > 4 && LocalPosition.at(nextnumber - 1)[0] < 7)
					{
						//float cofficient = (-LocalPosition.at(nextnumber - 1)[1]) / (LocalPosition.at(nextnumber - 1)[0] - 3.5);
						std::cout << "ffffffffffffffffffffff" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle)
						{
							client.moveByAngleThrottle(0.1,0, 0.6, 0.0f, 1.9f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.9f));
							last_circle = true;
							last_circle_weight = 0.0;
							client.hover();
						}
						else
						{
							if (!circle_middle_flag)
							{
								client.moveByAngleThrottle(-0.1*last_circle_weight, 0, 0.6, 0.0f, 1.75f);
								std::this_thread::sleep_for(std::chrono::duration<double>(1.75f));
								circle_middle_flag = true;
								client.hover();
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(0, -(y*0.01), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}

					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 10 && (LocalPosition.at(nextnumber - 1)[0]) >7)
					{
						std::cout << "33333333333333333333333" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle)
						{
							client.moveByAngleThrottle(0.1, 0, 0.61, 0.0f, 1.9f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.9f));
							last_circle = true;
							last_circle_weight = 0.001;
							client.hover();
						}
						else
						{
							if (!circle_far7_flag)
							{
								if (nextnumber == 6)
								{
									client.moveByAngleThrottle(-0.06*last_circle_weight, 0, 0.6, 0.0f, 2.54f);
									std::this_thread::sleep_for(std::chrono::duration<double>(2.54f));
									circle_far7_flag = true;
									client.hover();
								}
								else if (nextnumber == 4)
								{
									client.moveByAngleThrottle(-0.040*last_circle_weight, 0, 0.6, 0.0f, 2.54f);
									std::this_thread::sleep_for(std::chrono::duration<double>(2.54f));
									circle_far7_flag = true;
									client.hover();
								}
								else
								{
									client.moveByAngleThrottle(-0.040*last_circle_weight, 0, 0.6, 0.0f, 2.54f);
									std::this_thread::sleep_for(std::chrono::duration<double>(2.54f));
									circle_far7_flag = true;
									client.hover();
								}
								
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(0, -(y*0.01), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}

					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 12 && (LocalPosition.at(nextnumber - 1)[0]) >10)
					{
						std::cout << "44444444444444444444" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle)
						{
							client.moveByAngleThrottle(0.09, 0, 0.615, 0.0f, 1.5f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.5f));
							last_circle = true;
							last_circle_weight = 0.01;
							client.hover();
							std::this_thread::sleep_for(std::chrono::duration<double>(1.0f));
						}
						else
						{
							if (!circle_far10_flag)
							{
								std::cout << "last_circle_weight: " << last_circle_weight << std::endl;
								client.moveByAngleThrottle(-0.09*last_circle_weight, 0, 0.6, 0.0f, 2.9f);
								std::this_thread::sleep_for(std::chrono::duration<double>(2.9f));
								circle_far10_flag = true;
								client.hover();
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								//std::cout << "y: " << y << std::endl;
								client.moveByAngleThrottle(0, -(y*0.01), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}

					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 13 && (LocalPosition.at(nextnumber - 1)[0]) >12)
					{
						std::cout << "66666666666666666666666" << std::endl;
						//if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle) //上一个还是障碍圈
						//{
						//	std::cout << "上一个还是障碍圈：" << std::endl;
						//	client.moveByAngleThrottle(0.1, 0, 0.615, 0.0f, 1.9f);
						//	std::this_thread::sleep_for(std::chrono::duration<double>(1.9f));
						//	last_circle = true;
						//	last_circle_weight = 0.01;
						//	client.hover();
						//}
						/*else
						{*/
							if (!circle_far12_flag)
							{
								client.moveByAngleThrottle(-0.07*last_circle_weight, 0, 0.6, 0.0f, 2.90f);
								std::this_thread::sleep_for(std::chrono::duration<double>(2.90f));
								circle_far12_flag = true;
								client.hover();

							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(0, -(y*0.03), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						//}

					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 14 && (LocalPosition.at(nextnumber - 1)[0]) >13)
					{
						std::cout << "77777777777777777777777777" << std::endl;

						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle) //上一个还是障碍圈
						{
							std::cout << "上一个还是障碍圈" << std::endl;
							client.moveByAngleThrottle(0.1, 0, 0.615, 0.0f, 1.9f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.9f));
							last_circle = true;
							last_circle_weight = 0.05;
							client.hover();

						}
						else
						{
							if (!circle_far13_flag)
							{
								client.moveByAngleThrottle(-0.08*last_circle_weight, 0, 0.6, 0.0f, 3.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(3.1f));
								circle_far13_flag = true;
								client.hover();

							}
							else
							{

								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(-0.01, -(y*0.015), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}
					else if (LocalPosition.at(nextnumber - 1)[0] < 16 && (LocalPosition.at(nextnumber - 1)[0]) >14)
					{
						std::cout << "qqqqqqqqqqqqqqqqqqqqqqqqqq" << std::endl;

						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle) //上一个还是障碍圈
						{
							std::cout << "the next is queue" << endl;
							client.moveByAngleThrottle(0.1, 0, 0.615, 0.0f, 2.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
							last_circle = true;
							last_circle_weight = 0.05;
							client.hover();

						}
						else
						{
							if (!circle_far14_flag)
							{
								client.moveByAngleThrottle(-0.1*last_circle_weight, 0, 0.58, 0.0f, 4.00f);
								std::this_thread::sleep_for(std::chrono::milliseconds(4000));
								circle_far14_flag = true;
								client.hover();

							}
							else
							{

								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(0, -(y*0.015), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}

					else if (LocalPosition.at(nextnumber - 1)[0] < 18 && (LocalPosition.at(nextnumber - 1)[0]) >16)
					{
						std::cout << "8888888888888888888888888888" << std::endl;
						if (LocalPosition.at(nextnumber - 2)[2] < 0.5 && !last_circle) //上一个还是障碍圈
						{
							std::cout << "the next is queue" << std::endl;
							client.moveByAngleThrottle(0.1, 0.01, 0.615, 0.0f, 2.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(2.0f));
							last_circle = true;
							last_circle_weight = 0.05;
						}
						else
						{
							if (!circle_far18_flag)
							{
								client.moveByAngleThrottle(-0.08*last_circle_weight, 0, 0.6, 0.0f, 3.6f);
								std::this_thread::sleep_for(std::chrono::duration<double>(3.6f));
								circle_far18_flag = true;
								client.hover();
							}
							else
							{
								float y = LocalPosition.at(nextnumber - 1)[1];
								client.moveByAngleThrottle(-0.01, -(y*0.015), delta_throttle, 0.0f, 0.1f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
							}
						}
					}
					else
					{
						std::cout << "something went wrong! " << std::endl;
						client.moveByAngleThrottle(-0.05, 0, delta_throttle, 0.0f, 0.15f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
					}
					std::cout << "last_circle_weight: " << last_circle_weight << std::endl;
					 Min_distance = response.at(0).image_data_float.at(0);
					float pos_circle;
					for (int iter_num = 1; iter_num < response.at(0).image_data_float.size(); iter_num++)
					{
						if (response.at(0).image_data_float.at(iter_num) == 255) continue;
						if (Min_distance > response.at(0).image_data_float.at(iter_num))
						{
							Min_distance = response.at(0).image_data_float.at(iter_num);
							pos_circle = iter_num;
						}

					}
					if (Min_distance < 1.5)
					{
						std::cout << "back ........................" << std::endl;
						client.moveByAngleThrottle(0.09f, 0.0f, 0.60, 0.0f, 0.5f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.5f));
						
					}
					if (pos_circle > 640 * 240 && Min_distance < 2.5)
					{
						std::cout << " upup..........." << std::endl;
						client.moveByAngleThrottle(0.0f, 0.0f, 0.60, 0.0f, 0.2f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.2f));
					}
					else if (Min_distance < 2.5 && pos_circle < 640 * 240)
					{
						std::cout << " downdown..........." << std::endl;
						client.moveByAngleThrottle(0.0f, 0.0f, 0.55, 0.0f, 0.2f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.2f));
					}
				}
			}
			// 采图返回模式
			else if (controlmode == 4)
			{
				/************************** for collect data  ******************************/
				int photonum = 10;
				curr_bardata = client.getBarometerdata();
				ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
				ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);    //300多行的，三轴坐标，QR去年设好的，kalman滤波估计高度
				std::cout << "ned_curr: " << ned_curr(2) << std::endl;
				float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;      //推力控制量，//控制螺旋桨前后推力
				std::cout << "delta_throttle: " << delta_throttle << std::endl;
				CLIP3(0.4, delta_throttle, 0.8);//取饱和值，取（0.4,0.8之间）
				client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);//变成了阻塞的线程，不科学，应该是多线程并行
				std::this_thread::sleep_for(std::chrono::milliseconds(10));//动一下停一下
				int i;
				if (abs(ned_target(2) - ned_curr(2)) < 0.08)//设定的目标高度小于当前的目标高度0.08，不断让目标靠近
				{
					count4++;
				}
				if (count4 > 40)//有四十次和目标高度很相近
				{
					client.moveByAngleThrottle(0.0f, 0.1f, 0.59, 0.0f, 3.50f);//roll向右飞一段，取高度为25m，采样张数为n张
					std::this_thread::sleep_for(std::chrono::milliseconds(3500));//
					client.hover();//悬停，不然下一步转换方向前进不稳
					std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));

					i = photonum;
					for (i = 0; i < photonum; i++)//存图
					{
						sprintf(filename, "%d.png", i);
						flag_collect_data = false;
						while (!flag_collect_data)
						{
							data_circle planego;
							curr_bardata = client.getBarometerdata();//定高度，要保证高度相同，才能拼成一整张图，
							ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
							ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
							//pid_zerror << ned_curr(2) << endl;
							std::cout << "ned_curr: " << ned_curr(2) << std::endl;
							if (abs(ned_target(2) - ned_curr(2)) < 0.2) count_collect_data++;
							if (count_collect_data > 20)
							{
								planego.start(realdelta_x[i], 0);//设置需要的delta_x
								xcontrol(client, 1, planego.pitch, 0, planego.time_p);//根据计算出来的移动到需要的位置
								client.hover();
								xstay(client, 1.0);
								client.hover();
								clock_t start = clock();
								image = img2mat.get_below_mat();
								imwrite(filename, image);
								ImageForPosition.push_back(image);
								flag_collect_data = true;
							}
							else
							{
								//调整
								float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
								std::cout << "delta_throttle: " << delta_throttle << std::endl;
								CLIP3(0.4, delta_throttle, 0.8);
								client.moveByAngleThrottle(0.0f, 0.0f, (float)delta_throttle, 0.0f, 0.01f);
								std::this_thread::sleep_for(std::chrono::duration<double>(0.01f));
							}
						}
						count_collect_data = 0;
					}

					client.hover();
				}

				if (i == photonum)//采够n张，
				{

					for (int u = 0; u < 10; u++)
					{
						cv::Mat pic1, pic2;
						pic1 = ImageForPosition.at(u);
						pic1.copyTo(pic2);//拷贝pic的数据区到image中
						line(pic2, Point(1, 160), Point(640, 160), Scalar(0, 0, 255), 1, CV_AA);
						line(pic2, Point(1, 320), Point(640, 320), Scalar(0, 0, 255), 1, CV_AA);
						line(pic2, Point(map_pic_split[u][0], 1), Point(map_pic_split[u][0], 480), Scalar(0, 0, 255), 1, CV_AA);
						line(pic2, Point(map_pic_split[u][1], 1), Point(map_pic_split[u][1], 480), Scalar(0, 0, 255), 1, CV_AA);
						std::vector<Vec4i> range;
						std::vector<Vec3i> circles = detect_circle(pic2, range, u);//打算在这一步之后直接计算每一张图片rela_distance
						cv::Mat binImage_h;
						if (u == 0)
						{
							int res = pos_detect(pic1, binImage_h, u, 0);
							posarray[u] = res;//写入判断数组
						}
						else
						{
							int res = pos_detect(pic1, binImage_h, u, posarray[u - 1]);
							posarray[u] = res;//写入判断数组
						}
					}

					//修正前
					cout << "Before being corrected" << endl;
					for (int u = 0; u < 10; u++)
						std::cout << posarray[u];

					std::cout << endl;
					predict_check_position[0] = 0;

					for (int u = 1; u < 10; u++)
					{
						int lastpos = posarray[u - 1];
						int p = EzMap::DetectCircle_BaseOn_UpDown(ImageForPosition, u, lastpos);
						predict_check_position[u] = p;
						if (p != posarray[u])
						{
							doubt_array[u] = 1;
						}
						if (posarray[u] == -1)
						{
							doubt_array[u] = 1;
							posarray[u] = p;
						}
					}

					//修正后
					cout << "posarray" << endl;
					for (int u = 0; u < 10; u++)
						std::cout << posarray[u];
					std::cout << endl;

					cout << "predict_check_position" << endl;
					for (int u = 0; u < 10; u++)
						std::cout << predict_check_position[u];
					std::cout << endl;


					//准备回到停机坪，以初始姿态
					client.moveByAngleThrottle(0.0f, -0.1f, 0.59, 0.0f, 3.00f);//roll向左飞一段，取高度为height(m)，采样张数为n张
					std::this_thread::sleep_for(std::chrono::milliseconds(3000));//
					client.hover();//悬停，不然下一步转换方向前进不稳
					std::this_thread::sleep_for(std::chrono::duration<double>(1.0f));
					client.moveByAngleThrottle(0.1f, 0.0f, 0.59, 0.0f, 17.3f);//向后飞
					std::this_thread::sleep_for(std::chrono::milliseconds(16500));//这个前进长度刚好,不到一点点
					client.hover();//悬停
					std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));
					pillar_height(6, prev, client, Barometer_origin, pidZ);
					client.hover();//悬停
					std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));
					//以上飞到停机坪附近，应该是不到一些
					//pillar_height(12, prev, client, Barometer_origin, pidZ);

					//拍张照片看看
					count4 = 0;
					controlmode = 13;//钻圈模式
					test_delta_pitch = 0;

					// 计算并输出位置

				}
			}
			else if (controlmode == 5)
			{
				std::cout << " collect data complete! " << std::endl;
				if (!FLAG_CB)
				{
					if (flag_dis)
					{
						//LocalPosition = calculate(17.2, ImageForPosition, flag_con);//拼接，并处理 输出：x y flag(0,1)，
						cout << "\noutput the result" << endl;
					}
					// else

					// {
					// 	LocalPosition = LocalPosition_temp;
					// }
					// ###question: 这个移动是干嘛的？
					client.moveByAngleThrottle(-0.05, -0.1, 0.59, 0.0f, 2.2f);
					std::this_thread::sleep_for(std::chrono::milliseconds(2200));
					client.hover();
					std::this_thread::sleep_for(std::chrono::duration<double>(1.0f));
					FLAG_CB = true;
					cout << "\n--------------------------------------\n" << "the mode = 1\n" << "------------------------------------\n" << endl;
					controlmode = 11;
					count_home = 0;
					ned_target(2) = 7;
					pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
					flag_image = true;
					// ################## Attention this is break for test. if use , delete it
					/*TestRedAreaDetect();
					cout << "press the enter to get out" << endl;
					cin.get();
					break;*/
				}
				//	if (abs(ned_curr(2)- home_ned(2))<0.3)
				//	{
				//	//std::this_thread::sleep_for(std::chrono::duration<double>(5.0f));
				//	std::cout << " LocalPositionx: " << LocalPosition.at(nextnumber - 1)[0] << " LocalPositiony: " << LocalPosition.at(nextnumber - 1)[1] << std::endl;
				//	controlmode = 1;
				//	count_home = 0;
				//	ned_target(2) = 7;
				//	pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
				//	flag_image = true;
				//    }
				//else
				//{
				//	curr_bardata = client.getBarometerdata();
				//	ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
				//	ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
				//	
				//	std::cout <<" ned_curr(2): " << ned_curr(2) << std::endl;
				//	float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
				//	std::cout <<" delta_throttle: " << delta_throttle << std::endl;
				//	CLIP3(0.4, delta_throttle, 0.8);
				//	client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.0001f);
				//	std::this_thread::sleep_for(std::chrono::duration<double>(0.0001));
				//}
			}
			else if (controlmode == 6)
			{
				if (abs(ArucoBegin(2) - ned_curr(2)) < 0.3)
				{
					count_code++;
				}
				if (count_code > 50)
				{
					std::cout << "yaw adjust....." << std::endl;
					Magdata = client.getMagnetometerdata();
					float Mag_y = Magdata.magnetic_field_body.y();
					float yaw = pid_yaw.control(Mag_y);
					std::cout << " yaw: " << yaw << " Mag_y: " << Mag_y << std::endl;
					client.moveByAngleThrottle(0.0f, 0.0f, 0.58999, yaw, 0.01f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.01f));
					std::cout << "count_yaw: " << count_yaw << std::endl;
					if (abs(Mag_y - target_Mag_y) < 0.008) count_yaw++;
					if (count_yaw > 10 && abs(Mag_y - target_Mag_y) < 0.008)
					{
						
						client.moveByAngleThrottle(0.2, -0.08, 0.5888, 0.0f, 1.3f);
						std::this_thread::sleep_for(std::chrono::duration<double>(1.3));
						client.hover();
						std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
						if (!ReadTxt)
						{
							
							ReadTxt = true;
							controlmode = 7;
							count_code = 0;
							count_yaw = 0;						
							ned_target(0) = 100;
							ned_target(1) = 320;
							pidP_X.setPoint(ned_target(0), 0.0008, 0, 0.0005);// 这里的x指的是以无人机运动方向为x的反方向
							pidP_Y.setPoint(ned_target(1), 0.0008, 0, 0.0005);
						}						
					}
					else
					{
						curr_bardata = client.getBarometerdata();
						ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
						ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);						
						float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
						CLIP3(0.4, delta_throttle, 0.8);
						std::cout << "ned_curr(2): " << ned_curr(2) << std::endl;
						client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.01f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
					}
				}
				else
				{
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					//std::cout << " ned_curr(0): " << ned_curr(0) << " ned_curr(1): " << ned_curr(1) << std::endl;
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;

					CLIP3(0.4, delta_throttle, 0.8);
					//std::cout << "test_delta_pitch: " << test_delta_pitch << " delta_roll: " << delta_roll << std::endl;
					client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.01f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
				}
			}
			else if (controlmode == 7)           //finding the target tree
			{
				////需要先飞到停机坪
				while (1)
				{
					//platform_move();
				}
				std::cout << "TREE_NUM: " << TREE_NUM << std::endl;
							
				Img = img2mat.get_below_mat();
				
				if (!Img.empty())
				{
					switch (state)
					{
					case left2right:
						//iter_num = 27;
						rollCoefficient = 1;
						//pitchCoefficient = 0;
						//state = forward;
						std::cout << "state: " << "left2right" << std::endl;
						break;
					
					case right2left:						
						rollCoefficient = -1;
						//pitchCoefficient = 0;
						//state = forward;
						std::cout << "state: " << "right2left" << std::endl;
						statechange = true;
						break;
					default:
						iter_num = 0;
						rollCoefficient = 0;
						pitchCoefficient = 0;
						std::cout << "default " << std::endl;
						break;
					}
					std::vector<Vec2i> tree_circles = detect_tree(Img, ned_curr(2));
					std::cout << "the tree size: " << tree_circles.size() << endl;
					if (tree_circles.size() > 0 )
					{
						if (tree_circles.size() == 1)
						{
							int MAX = tree_circles.at(0)[1];
							XY_TREE = tree_circles.at(0);
							for (int i = 0; i < tree_circles.size(); i++)
							{
								if (tree_circles.at(i)[1] > MAX)
								{
									MAX = tree_circles.at(i)[1];
									XY_TREE = tree_circles.at(i);
								}

							}
						}
						else
						{
							int Min = tree_circles.at(0)[0];
							XY_TREE = tree_circles.at(0);
							for (int i = 0; i < tree_circles.size(); i++)
							{
								if (tree_circles.at(i)[0] < Min)
								{
									Min = tree_circles.at(i)[0];
									XY_TREE = tree_circles.at(i);
								}

							}
						}
						
						if (XY_TREE[0] == 0 && XY_TREE[1] == 0)
						{
							std::cout << "no full circle detected......" << std::endl;							
							if (TREE_NUM > 2 && TREE_NUM<7) state = left2right;
							else state = right2left;
							client.moveByAngleThrottle(-0.03, 0.03*rollCoefficient, 0.593333, 0.05f, 0.001f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
						}
						else
						{
							curr_bardata = client.getBarometerdata();
							ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
							ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);							
							std::cout << "XY_TREE[0]: " << XY_TREE[0] << " XY_TREE[1]: " << XY_TREE[1] << std::endl;
							float pitch = pidP_X.control(XY_TREE[0]);
							float roll = pidP_Y.control(XY_TREE[1]);
							float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
							Magdata = client.getMagnetometerdata();
							float Mag_y = Magdata.magnetic_field_body.y();
							float yaw = pid_yaw.control(Mag_y);
							CLIP3(-0.1, pitch, 0.1);
							CLIP3(-0.1, roll, 0.1);
							CLIP3(0.4, delta_throttle, 0.8);
							std::cout << "pitch: " << pitch << "  roll: " << roll << std::endl;
							client.moveByAngleThrottle(-pitch, -roll, delta_throttle, yaw, 0.1f);
							std::this_thread::sleep_for(std::chrono::duration<double>(0.1f));
						}

					}
					else
					{
						
						std::cout << "no full circle detected......" << std::endl;
						
						if (TREE_NUM > 2 && TREE_NUM<8) state = left2right;
						else state = right2left;
						client.moveByAngleThrottle(-0.03, 0.02*rollCoefficient, 0.59333, 0.f, 0.001f);
						std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
					}
					if (abs(XY_TREE[0] - 100) < 20 && abs(XY_TREE[1] - 320) < 20) TREE_COUNT++;
					if (TREE_COUNT > 0)
					{
						
						controlmode = 8;                  // down and then scan ARuco code
						TREE_COUNT = 0;
						XY_TREE[0] = 0;
						XY_TREE[1] = 0;	
						ned_target(2) = 3.5;
						pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
						std::cout << "controlmode turn to 8....." << std::endl;
						client.hover();
						std::this_thread::sleep_for(std::chrono::duration<double>(2.0));
					}
				}
			
			}
			else if (controlmode == 8)   // down and then scan ARuco code
			{
				if (countResult == 5)
				{
					controlmode = 9;
					std::cout << "Aruco code collect end! go to 0. " << std::endl;
					ned_target(2) = 9;
					pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);

				}
				if (FLAG_UPDOWN && !is_FLAG_DOWN)  //FLAG_UPDOWN: true, down ;false, up
				{
					std::cout << "down down .............." << std::endl;
					ned_target(2) = 3.5;
					pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
					is_FLAG_DOWN = true;
				}
				else if(!FLAG_UPDOWN && !is_FLAG_UP)
				{
					std::cout << "up up .............." << std::endl;
					ned_target(2) = 11;
					pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
					is_FLAG_UP = true;
				}
				else
				{
					std::cout << "emmmmmmm .............." << std::endl;
				}				
				if (abs(ned_target(2) - ned_curr(2)) < 0.2) UPDOWN_COUNT++;
				std::cout << "UPDOWN_COUNT: " << UPDOWN_COUNT << std::endl;
				if (UPDOWN_COUNT > 10)
				{					
					if (FLAG_UPDOWN)   //FLAG_UPDOWN: true, down ;false, up
					{
						client.hover();
						std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
						cv::Mat imgCode = img2mat.get_front_mat();						
						imwrite("aruco.png", imgCode);
						std::vector< int > ids;
						std::vector<std::vector< Point2f > > corners, rejected;
						std::vector< Vec3d > rvecs, tvecs;
						//detect markers and estimate pose
						aruco::detectMarkers(imgCode, dictionary, corners, ids, detectorParams, rejected);
						if (ids.size() > 0)
						{
							aruco::drawDetectedMarkers(imgCode, corners, ids);
							//imshow("out", imgCode);
							//cv::waitKey(1);
							for (int ids_iter = 0; ids_iter < ids.size(); ids_iter++)
							{
								for (int aruco_iter = 0; aruco_iter < 10; aruco_iter++)
								{
									if (ids.at(ids_iter) == ArucoID[aruco_iter])
									{
										if (result[aruco_iter] == 1)
										{
											std::vector<ImageResponse> responseAruco = img2mat.get_depth_mat();
											SaveResult(imgCode, responseAruco.at(0), corners.at(ids_iter), ArucoID[aruco_iter]);
											result[aruco_iter] = 0;
											countResult++;
											break;
										}										
										
										//break;
									}									
								}
							}
							
							ned_target(2) = 11;
							pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
							UPDOWN_COUNT = 0;
							TREE_NUM = TREE_NUM + 1;
							FLAG_UPDOWN = false;
							//is_FLAG_UPDOWN = false;
							is_FLAG_UP = false;
							

						}
						else
						{
							ned_target(2) = 11;
							pidZ.setPoint(ned_target(2), 0.3, 0, 0.4);
							UPDOWN_COUNT = 0;
							TREE_NUM = TREE_NUM + 1;
							is_FLAG_UP = false;
							FLAG_UPDOWN = false;
							//is_FLAG_UPDOWN = false;
						}
					}                   
					else                     //up and go for next code
					{
						std::cout << "go for next ARuco code......" << std::endl;
						std::cout << "TREE_NUM: " << TREE_NUM<<std::endl;
						if (TREE_NUM == 3 || TREE_NUM == 8)
						{
							std::cout << "back a little......" << std::endl;
							client.moveByAngleThrottle(0.45, 0, 0.593333, 0.05f, 1.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
							client.hover();
							std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
							
						}
						else if (TREE_NUM<3 || TREE_NUM > 8)
						{
							client.moveByAngleThrottle(0, -0.3, 0.59, 0.0f, 1.5f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.8));
						}
						else if ((TREE_NUM > 3 && TREE_NUM < 6) || TREE_NUM==7)
						{
							client.moveByAngleThrottle(-0.08, 0.3, 0.59, 0.0f, 2.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.8));
						}
						else if(TREE_NUM==6)
						{
							client.moveByAngleThrottle(-0.3, 0.3, 0.59, 0.0f, 2.0f);
							std::this_thread::sleep_for(std::chrono::duration<double>(1.8));
						}
						else
						{
							controlmode = 9;
						}
						
						client.hover();
						std::this_thread::sleep_for(std::chrono::duration<double>(1.8));
						controlmode = 7;
						UPDOWN_COUNT = 0;						
						FLAG_UPDOWN = true;
						is_FLAG_DOWN = false;
						//TREE_NUM = TREE_NUM + 1;
						//is_FLAG_UPDOWN = false;
						
					}
				}
				else
				{
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
					CLIP3(0.4, delta_throttle, 0.8);
					std::cout << "delta_throttle: " << delta_throttle << std::endl;
					client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.01f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
				}
			}
			else if (controlmode == 9)
			{
				if (abs(ned_target(2) - ned_curr(2)) < 0.2) HOME_COUNT++;
			 //	std::cout << "controlmode: " << UPDOWN_COUNT << std::endl;
				if (HOME_COUNT > 10)
				{
					client.moveByAngleThrottle(-0.1, 0, 0.5899999, 0.0f, 17.5f);
					std::this_thread::sleep_for(std::chrono::duration<double>(17.5));
					client.hover();
					std::this_thread::sleep_for(std::chrono::duration<double>(5.0));
					controlmode = 1;					
					HOME = true;
					controlmode = 10;
					HOME_COUNT = 0;
				}
				else
				{
					curr_bardata = client.getBarometerdata();
					ned_curr(2) = curr_bardata.altitude - Barometer_origin.altitude;
					ned_curr(2) = kalman(ned_curr(2), prev, i_kalman, Q, R);
					std::cout << "ned_curr(2): " << ned_curr(2) << std::endl;
					float delta_throttle = pidZ.control(ned_curr(2)) + 0.6;
					CLIP3(0.4, delta_throttle, 0.8);
					std::cout << "delta_throttle: " << delta_throttle << std::endl;
					client.moveByAngleThrottle(0, 0, delta_throttle, 0.0f, 0.01f);
					std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
				}

			}	
			clock_t end = clock();
			std::cout << "cost time: " << end - begin << std::endl;
			long time_temp = 50 - (end - begin);
			//std::this_thread::sleep_for(std::chrono::milliseconds(time_temp));
			i_kalman = i_kalman + 1;
			x_kalman = x_kalman + 1;
			y_kalman = x_kalman + 1;
		}
	}
	catch (rpc::rpc_error&  e)
	{
		std::string msg = e.get_error().as<std::string>();
		std::cout << "Exception raised by the API, something went wrong." << std::endl << msg << std::endl;
	}

	return 0;
}


