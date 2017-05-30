#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/gpu/gpu.hpp"
using namespace cv;
using namespace std;
int main(int argc, char** argv){
	//std::cout << " hello opencv" << std::endl;
	cv::Mat mat = cv::imread(argv[1]);
	gpu::GpuMat src;
	Size ksize;
	ksize.width =3;
	ksize.height =3;
	cv::gpu::GpuMat gray;
	cv::gpu::GpuMat blurred;
	cv::gpu::GpuMat edges;
	Mat edges_host;
	double timeSec = 0;
    for(;;)
    {
	const int64 startWhole = getTickCount();	
	src.upload(mat);
	const int64 startCvt = getTickCount();
        cv::gpu::cvtColor(src, gray, CV_BGR2GRAY);
	timeSec = (getTickCount() - startCvt) / getTickFrequency();
	std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;

	const int64 startBox = getTickCount();
        gpu::boxFilter(gray, blurred, -1,ksize);
	timeSec = (getTickCount() - startBox) / getTickFrequency();
	std::cout << "		BoxFilter Time : " << timeSec << " sec" << std::endl;

	const int64 startCanny = getTickCount();
        gpu::Canny(blurred, edges, 120, 240, 3, false);	
	timeSec = (getTickCount() - startCanny) / getTickFrequency();
	std::cout << "		Canny Time : " << timeSec << " sec" << std::endl;

	edges.download(edges_host);
	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	std::cout << "	Process Time : " << timeSec << " sec" << std::endl;

        imshow("edges", edges_host);
	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	std::cout << "Total Time : " << timeSec << " sec" << std::endl;
        if(waitKey(30) >= 0) break;
    }
	return 0;
}
