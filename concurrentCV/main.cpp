#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv){
	//std::cout << " hello opencv" << std::endl;
	cv::Mat mat = cv::imread("512kb.jpg");
	Mat gray1;
	Mat gray2;
	for(int i=0; i<10000; i++){
		//std::cout << "hello opencv" << std::endl;
		cv::cvtColor(mat, gray1, CV_BGR2GRAY);
		cv::cvtColor(mat, gray2, CV_BGR2GRAY);
	}
	imwrite("result1.png", gray1);
	imwrite("result2.png", gray2);

	return 0;
}
