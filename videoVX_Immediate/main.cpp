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


#include <NVX/nvx_opencv_interop.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
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
	std::string input = "../NORWAY2K.mp4";
        //
        // Parse command line arguments
        //
	app.addOption('s', "source", "Input URI", nvxio::OptionHandler::string(&input));
	

        app.init(argc, argv);

        //
        // Create OpenVX context
        //

        nvxio::ContextGuard context;
	vxDirective(context, VX_DIRECTIVE_ENABLE_PERFORMANCE);
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

        std::unique_ptr<nvxio::Render> render(nvxio::createDefaultRender(
                    context, "Player Sample", config.frameWidth, config.frameHeight));
        if (!render)
        {
            std::cout << "Error: Cannot open default render!" << std::endl;
            return nvxio::Application::APP_EXIT_CODE_NO_RENDER;
        }

        EventData eventData;
        render->setOnKeyboardEventCallback(keyboardEventCallback, &eventData);

        vx_image frame = vxCreateImage(context, config.frameWidth,
                                       config.frameHeight, config.format);
        NVXIO_CHECK_REFERENCE(frame);
 
	//create vx objects

	vx_image gray = vxCreateImage(context, config.frameWidth,  config.frameHeight, VX_DF_IMAGE_U8);
	vx_image blurred = vxCreateImage(context, config.frameWidth,  config.frameHeight, VX_DF_IMAGE_U8);
	vx_image edges = vxCreateImage(context, config.frameWidth,  config.frameHeight, VX_DF_IMAGE_U8);
	vx_threshold CannyThreshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_INT32);
	int lowerTresh = 120;
	int upperTresh = 240;
	NVXIO_CHECK_REFERENCE(CannyThreshold);
	NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER,
					&lowerTresh, sizeof(lowerTresh)) );
	NVXIO_SAFE_CALL( vxSetThresholdAttribute(CannyThreshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER,
					&upperTresh, sizeof(upperTresh)) );


        nvxio::Render::TextBoxStyle style = {{255,255,255,255}, {0,0,0,127}, {10,10}};

        std::unique_ptr<nvxio::SyncTimer> syncTimer = nvxio::createSyncTimer();
        syncTimer->arm(1. / app.getFPSLimit());

	//set Total timer;
	double proc_ms = 0;
        nvx::Timer totalTimer;
        totalTimer.tic();
	double timeSec = 0;
        while(eventData.alive)
        {
            nvxio::FrameSource::FrameStatus status = nvxio::FrameSource::OK;
            if (!eventData.pause)
            {
                status = source->fetch(frame);
            }
	
		const int64 startWhole = getTickCount();
		const int64 startCvt = getTickCount();
		vxuColorConvert(context, frame, gray);
		timeSec = (getTickCount() - startCvt) / getTickFrequency();
		std::cout << "		Convert Time : " << timeSec << " sec" << std::endl;

		const int64 startBox = getTickCount();
		vxuBox3x3(context, gray, blurred);
		timeSec = (getTickCount() - startBox) / getTickFrequency();
		std::cout << "		BoxFilter Time : " << timeSec << " sec" << std::endl;

		const int64 startCanny = getTickCount();
		vxuCannyEdgeDetector(context, blurred, CannyThreshold, 3, VX_NORM_L1, edges);
		timeSec = (getTickCount() - startCanny) / getTickFrequency();
		std::cout << "		CannyFilter Time : " << timeSec << " sec" << std::endl;
 

		timeSec = (getTickCount() - startWhole) / getTickFrequency();
		std::cout << "	Process Time : " << timeSec << " sec" << std::endl; 		

            switch(status)
            {
            case nvxio::FrameSource::OK:
            {

                double total_ms = totalTimer.toc();


                std::cout << "Display Time : " << total_ms << " ms" << std::endl << std::endl;

                syncTimer->synchronize();

                total_ms = totalTimer.toc();

                totalTimer.tic();

               std::ostringstream txt;
                txt << std::fixed << std::setprecision(1);

                txt << "Source size: " << config.frameWidth << 'x' << config.frameHeight << std::endl;
                txt << "Algorithm: " << proc_ms << " ms / " << 1000.0 / proc_ms << " FPS" << std::endl;
                txt << "Display: " << total_ms  << " ms / " << 1000.0 / total_ms << " FPS" << std::endl;

                txt << std::setprecision(6);
                txt.unsetf(std::ios_base::floatfield);

                txt << "LIMITED TO " << app.getFPSLimit() << " FPS FOR DISPLAY" << std::endl;
                txt << "Space - pause/resume" << std::endl;
                txt << "Esc - close the demo";
		

                render->putImage(edges);
                render->putTextViewport(txt.str(), style);

                if (!render->flush())
                    eventData.alive = false;
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
        
	vxReleaseImage(&frame);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return nvxio::Application::APP_EXIT_CODE_ERROR;
    }

    return nvxio::Application::APP_EXIT_CODE_SUCCESS;
}
