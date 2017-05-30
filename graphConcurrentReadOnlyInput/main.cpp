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

	vx_context ctx = vxCreateContext();
	vx_graph graph = vxCreateGraph(ctx);
	vx_image image1 = vxCreateImage(ctx, 10, 10, VX_DF_IMAGE_U8);
	vx_image image2 = vxCreateImage(ctx, 10, 10, VX_DF_IMAGE_U8);
	vx_lut lut = vxCreateLUT(ctx, VX_TYPE_UINT8, 256);
	vxTableLookupNode(graph, image1, lut, image2);
	vx_size size;
	vx_map_id map_id;
	void *ptr;
	// In thread 1:
   	 vxProcessGraph(graph);
	// In thread 2:
   	 vxQueryImage(image1, VX_IMAGE_ATTRIBUTE_SIZE, &size, sizeof size);
   	 vxMapLUT(lut, &map_id, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
   	 vxUnmapLUT(lut, map_id);
	// Valid: both `image1` and `lut` are only used as inputs by the graph, and
	// concurrent uses are limited to reads and read-only mappings.


        return nvxio::Application::APP_EXIT_CODE_SUCCESS;
    }
