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
        //std::unique_ptr<nvxio::Render> render(nvxio::createDefaultRender(
                    //context, "Player Sample", width, height));	

       // if (!render) {
       //     std::cerr << "Error: Can't create a renderer." << std::endl;
       //     return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
       // }
	//create virtual image 
	        
	vx_graph graph = vxCreateGraph(context);
	vx_image gray = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        vx_image blurred = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        vx_image edges = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);       
	vx_image result = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
	vx_threshold CannyThreshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_INT32);
	int lowerTresh = 120;
	int upperTresh = 240;
	NVXIO_CHECK_REFERENCE(CannyThreshold);
	NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,
					&lowerTresh, sizeof(lowerTresh)) );
	NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,
					&upperTresh, sizeof(upperTresh)) );


 	// node creation
	vx_node cvtNode = vxColorConvertNode(graph, src, gray);
	NVXIO_CHECK_REFERENCE(cvtNode);
	vx_node boxNode = vxBox3x3Node(graph, gray, blurred);
	NVXIO_CHECK_REFERENCE(boxNode);
	vx_node cannyNode = vxCannyEdgeDetectorNode(graph, blurred, CannyThreshold, 3, VX_NORM_L1, edges);
	NVXIO_CHECK_REFERENCE(cannyNode);

	//Ensure highest graph optimization level 
	const char* option = "-O3";
	vxSetGraphAttribute(graph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)); 

        //nvxio::Render::TextBoxStyle style = {{255,255,255,255}, {0,0,0,127}, {10,10}};


	//set Total timer;
	double proc_ms = 0;
        nvx::Timer totalTimer;
        totalTimer.tic();


        while(true)
        {
 
		//process graph	
        	nvx::Timer procTimer;
        	procTimer.tic();
	
		NVXIO_SAFE_CALL( vxProcessGraph(graph) );
		proc_ms = procTimer.toc();
		vx_perf_t perf;
		//query graph
		NVXIO_SAFE_CALL( vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
		std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms:" << proc_ms << std::endl;

		//query node
		NVXIO_SAFE_CALL( vxQueryNode(cvtNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
		std::cout << "\t Color Convert Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

		NVXIO_SAFE_CALL( vxQueryNode(boxNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
		std::cout << "\t BoxxTime : " << perf.tmp / 1000000.0 << " ms" << std::endl;


		NVXIO_SAFE_CALL( vxQueryNode(cannyNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
		std::cout << "\t Canny Time : " << perf.tmp / 1000000.0 << " ms" << std::endl; 


                double total_ms = totalTimer.toc();


                std::cout << "Display Time : " << total_ms << " ms" << std::endl << std::endl;



                total_ms = totalTimer.toc();

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

                //render->putImage(edges);
                //render->putTextViewport(txt.str(), style);
		//render->flush();
        }


        vxReleaseImage(&src);
        vxReleaseImage(&gray);
	vxReleaseNode(&cvtNode);
	vxReleaseNode(&boxNode);
	vxReleaseNode(&cannyNode);
	vxReleaseGraph(&graph);



        return nvxio::Application::APP_EXIT_CODE_SUCCESS;
    }
