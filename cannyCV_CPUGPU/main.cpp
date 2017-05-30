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

	cv::gpu::CudaMem host_src_pl(720, 1080, CV_8UC3, cv::gpu::CudaMem::ALLOC_PAGE_LOCKED);
	cv::gpu::CudaMem host_dst_pl;

	Mat mat = host_src_pl;
	mat = cv::imread(argv[1]);
	gpu::GpuMat src;
	Size ksize;
	ksize.width =3;
	ksize.height =3;

	cv::gpu::Stream stream;


	cv::gpu::GpuMat gray;
	cv::gpu::GpuMat blurred;
	cv::gpu::GpuMat edges;


	cv::Mat gray2;
	cv::Mat blurred2;
	cv::Mat edges2;

	Mat edges_host;
	double timeSec = 0;
	double averageTime = 0;
    for(int i=0; i<100; i++)
    {
	const int64 startWhole = getTickCount();	
	//src.upload(mat);
	stream.enqueueUpload(host_src_pl, src);
	stream.waitForCompletion();
	const int64 startCvt = getTickCount();
        cv::gpu::cvtColor(src, gray, CV_BGR2GRAY,0, stream);
	timeSec = (getTickCount() - startCvt) / getTickFrequency();
	std::cout << "		Convert GPU Time : " << timeSec << " sec" << std::endl;



	const int64 startBox = getTickCount();
        gpu::boxFilter(gray, blurred, -1,ksize, Point(-1, -1), stream);
	timeSec = (getTickCount() - startBox) / getTickFrequency();
	std::cout << "		BoxFilter GPU Time : " << timeSec << " sec" << std::endl;




	//const int64 startCanny = getTickCount();
        //gpu::Canny(blurred, edges, 120, 240, 3, false);	
	//timeSec = (getTickCount() - startCanny) / getTickFrequency();
	//std::cout << "		Canny GPU Time : " << timeSec << " sec" << std::endl;



	edges.download(edges_host);
	stream.enqueueDownload(blurred, host_dst_pl);

	//const int64 startCvtCpu = getTickCount();
         cv::cvtColor(mat, gray2, CV_BGR2GRAY);
	//timeSec = (getTickCount() - startCvtCpu) / getTickFrequency();
	//std::cout << "		Convert CPU Time : " << timeSec << " sec" << std::endl;

	//const int64 startBoxCpu = getTickCount();
        boxFilter(gray2, blurred2, -1,ksize);
	//timeSec = (getTickCount() - startBoxCpu) / getTickFrequency();
	//std::cout << "		BoxFilter CPU Time : " << timeSec << " sec" << std::endl;

	//const int64 startCannyCpu = getTickCount();
        //Canny(blurred2, edges2, 120, 240, 3);	
	//timeSec = (getTickCount() - startCannyCpu) / getTickFrequency();
	//std::cout << "		Canny CPU Time : " << timeSec << " sec" << std::endl;
	 stream.WaitCompletion();
	
	//edges_host = host_dst_pl;
	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	std::cout << "	Process Time : " << timeSec << " sec" << std::endl;
	if(i >= 1){
		averageTime += timeSec;	
	}
        //imshow("edges", edges_host);
	timeSec = (getTickCount() - startWhole) / getTickFrequency();
	std::cout << "Total Time : " << timeSec << " sec" << std::endl;
        if(waitKey(30) >= 0) break;
    }
	std::cout << "Average Time : " << averageTime/99 << " sec" << std::endl;
	return 0;
}
