#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>

#include <VX/vx.h>
#include <NVX/nvx_timer.hpp>

#include "NVXIO/Application.hpp"
#include "NVXIO/FrameSource.hpp"
#include <NVXIO/ConfigParser.hpp>
#include "NVXIO/Render.hpp"
#include "NVXIO/SyncTimer.hpp"
#include "NVXIO/Utility.hpp"

#include <NVX/nvx_opencv_interop.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
    int main(int argc, char** argv)
    {

	cv::Mat mat = cv::imread(argv[1]);
        int width = mat.cols;
        int height = mat.rows;
        nvxio::ContextGuard context;
	vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);
        vx_image src =  nvx_cv::createVXImageFromCVMat(context, mat);

	int a = 0;

	//create a render
        std::unique_ptr<nvxio::Render> render(nvxio::createDefaultRender(
                    context, "Player Sample", width, height));	

       if (!render) {
            std::cerr << "Error: Can't create a renderer." << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }
	//create images 
	        
	vx_image gray = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        vx_image blurred = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        vx_image edges = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);       
	vx_image result = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
	vx_threshold CannyThreshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_INT32);
	int lowerTresh = 120;
	int upperTresh = 240;
	NVXIO_CHECK_REFERENCE(CannyThreshold);
	NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,
					&lowerTresh, sizeof(lowerTresh)) );
	NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,
					&upperTresh, sizeof(upperTresh)) );

        nvxio::Render::TextBoxStyle style = {{255,255,255,255}, {0,0,0,127}, {10,10}};


	//set Total timer;
	double proc_ms = 0;
        nvx::Timer totalTimer;
        totalTimer.tic();

	double timeSec = 0;
        while(true)
        {
		const int64 startWhole = getTickCount();
		const int64 startCvt = getTickCount();
		vxuColorConvert(context, src, gray);
		timeSec = (getTickCount() - startCvt) / getTickFrequency();
		std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;

		const int64 startBox = getTickCount();
		vxuBox3x3(context, gray, blurred);
		timeSec = (getTickCount() - startBox) / getTickFrequency();
		std::cout << "		BoxFilter Time : " << timeSec << " sec" << std::endl;

		const int64 startCanny = getTickCount();
		vxuCannyEdgeDetector(context, blurred, CannyThreshold, 3, VX_NORM_L1, edges);
		timeSec = (getTickCount() - startCanny) / getTickFrequency();
		std::cout << "		Canny Time : " << timeSec << " sec" << std::endl;
 

		timeSec = (getTickCount() - startWhole) / getTickFrequency();
		std::cout << "	Process Time : " << timeSec << " sec" << std::endl; 
             
		double total_ms = totalTimer.toc();

                totalTimer.tic();
                totalTimer.tic();

                std::ostringstream txt;
                txt << std::fixed << std::setprecision(1);

                txt << "Source size: " << width << 'x' << height << std::endl;
                txt << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
                txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

                txt << std::setprecision(6);
                txt.unsetf(std::ios_base::floatfield);

                txt << "LIMITED TO " << 29.97 << " FPS FOR DISPLAY" << std::endl;
                txt << "Space - pause/resume" << std::endl;
                txt << "Esc - close the demo";
		//NVXIO_SAFE_CALL( nvxuCopyImage(context, edges, result) );

                render->putImage(edges);
                render->putTextViewport(txt.str(), style);
		render->flush();
        }


        vxReleaseImage(&src);
        vxReleaseImage(&gray);

        return nvxio::Application::APP_EXIT_CODE_SUCCESS;
    }
