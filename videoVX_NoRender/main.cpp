/*
# Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

struct EventData
{
    EventData(): alive(true), pause(false) {}

    bool alive;
    bool pause;
};

static void keyboardEventCallback(void* context, vx_char key, vx_uint32 /*x*/, vx_uint32 /*y*/)
{
    EventData* eventData = static_cast<EventData*>(context);
    if (key == 27) // escape
    {
        eventData->alive = false;
    }
    else if (key == 32)
    {
        eventData->pause = !eventData->pause;
    }
}

//
// main - Application entry point
//

int main(int argc, char** argv)
{
    try
    {
        nvxio::Application &app = nvxio::Application::get();

        //
        // Parse command line arguments
        //

        std::string input = "NORWAY 2K.mp4";

        app.init(argc, argv);

        //
        // Create OpenVX context
        //

        nvxio::ContextGuard context;

        //
        // Messages generated by the OpenVX framework will be processed by nvxio::stdoutLogCallback
        //

        //vxRegisterLogCallback(context, &nvxio::stdoutLogCallback, vx_false_e);

        //
        // Create a Frame Source
        //

        std::unique_ptr<nvxio::FrameSource> source(nvxio::createDefaultFrameSource(context, input));
        if (!source || !source->open())
        {
            std::cerr << "Error: Can't open source URI " << input << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RESOURCE;
        }
        nvxio::FrameSource::Parameters config = source->getConfiguration();




	 //
        // Create a Render
        //

        //std::unique_ptr<nvxio::Render> render(nvxio::createDefaultRender(
        //            context, "Player Sample", config.frameWidth, config.frameHeight));
        //if (!render)
        //{
        //    std::cout << "Error: Cannot open default render!" << std::endl;
        //    return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        //}

        EventData eventData;
        //render->setOnKeyboardEventCallback(keyboardEventCallback, &eventData);

        vx_image frame = vxCreateImage(context, config.frameWidth,
                                       config.frameHeight, config.format);
        NVXIO_CHECK_REFERENCE(frame);
 
	//create vx objects
	vx_graph graph = vxCreateGraph(context);
	vx_image gray = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
	vx_image blurred = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
	vx_image edges = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
	vx_threshold CannyThreshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_INT32);
	int lowerTresh = 120;
	int upperTresh = 240;
	NVXIO_CHECK_REFERENCE(CannyThreshold);
	NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,
					&lowerTresh, sizeof(lowerTresh)) );
	NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,
					&upperTresh, sizeof(upperTresh)) );

	// node creation
	vx_node cvtNode = vxColorConvertNode(graph, frame, gray);
	vx_node boxNode = vxBox3x3Node(graph, gray, blurred);
	vx_node cannyNode = vxCannyEdgeDetectorNode(graph, blurred, CannyThreshold, 3, VX_NORM_L1, edges);



	//Ensure highest graph optimization level 
	const char* option = "-O3";
	vxSetGraphAttribute(graph, NVX_GRAPH_VERIFY_OPTIONS, option, strlen(option)); 

        //nvxio::Render::TextBoxStyle style = {{255,255,255,255}, {0,0,0,127}, {10,10}};

        std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
        syncTimer->arm(1. / app.getFPSLimit());

        nvx::Timer totalTimer;
        totalTimer.tic();
        while(eventData.alive)
        {
            nvxio::FrameSource::FrameStatus status = nvxio::FrameSource::OK;
            if (!eventData.pause)
            {
                status = source->fetch(frame);
            }



		//process graph		
		vxProcessGraph(graph);






            switch(status)
            {
            case nvxio::FrameSource::OK:
            {

                double total_ms = totalTimer.toc();

                std::cout << "NO PROCESSING" << std::endl;
                std::cout << "Display Time : " << total_ms << " ms" << std::endl << std::endl;

                syncTimer->synchronize();

                total_ms = totalTimer.toc();

                totalTimer.tic();




               std::ostringstream txt;
                txt << std::fixed << std::setprecision(1);

                txt << "Source size: " << config.frameWidth << 'x' << config.frameHeight << std::endl;
               txt << "Algorithm: " << "No Processing" << std::endl;
                txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

                txt << std::setprecision(6);
                txt.unsetf(std::ios_base::floatfield);

                txt << "LIMITED TO " << app.getFPSLimit() << " FPS FOR DISPLAY" << std::endl;
                txt << "Space - pause/resume" << std::endl;
                txt << "Esc - close the demo";
		vx_perf_t perf;
		//query graph
		// NVXIO_SAFE_CALL( vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
		//std::cout << "Graph Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

		//query node
		// NVXIO_SAFE_CALL( vxQueryNode(cvtNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
		//std::cout << "\t Color Convert Time : " << perf.tmp / 1000000.0 << " ms" << std::endl;

		// NVXIO_SAFE_CALL( vxQueryNode(boxNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
		//std::cout << "\t BoxxTime : " << perf.tmp / 1000000.0 << " ms" << std::endl;


		// NVXIO_SAFE_CALL( vxQueryNode(cannyNode, VX_NODE_ATTRIBUTE_PERFORMANCE, &perf, sizeof(perf)) );
		//std::cout << "\t Canny Time : " << perf.tmp / 1000000.0 << " ms" << std::endl; 
                //render->putImage(edges);
                //render->putTextViewport(txt.str(), style);

                //if (!render->flush())
                //    eventData.alive = false;
            } break;
            case nvxio::FrameSource::TIMEOUT:
            {
                // Do nothing
            } break;
            case nvxio::FrameSource::CLOSED:
            {
                // Reopen
                if (!source->open())
                {
                    std::cerr << "Error: Failed to reopen the source" << std::endl;
                    eventData.alive = false;
                }
            } break;
            }
        }

        //
        // Release all objects
        //
        
	vxReleaseNode(&cvtNode);
	vxReleaseNode(&boxNode);
	vxReleaseNode(&cannyNode);
	vxReleaseGraph(&graph);
	vxReleaseImage(&frame);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}