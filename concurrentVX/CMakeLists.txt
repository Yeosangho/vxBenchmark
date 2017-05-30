CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(concurrentVX)
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} PATH)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
SET(CMAKE_CXX_FLAGS "-std=c++11 -g -O3 -Wno-unused-result -Wno-deprecated -fPIC -Wno-deprecated-gpu-targets")
option(CUDA "Select to compile CUDA support by default if available" ON)
find_package(CUDA)
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11; --default-stream per-thread; -arch=sm_53;)
LIST(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)  
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(OpenCV REQUIRED core imgproc video highgui calib3d gpu)
find_package(VisionWorks REQUIRED)
find_package(VisionWorks-NVXIO)
include_directories("${OpenCV_INCLUDE_DIRS}")
include_directories("${VisionWorks_INCLUDE_DIRS}")
include_directories("${VisionWorks-NVXIO_INCLUDE_DIRS}")


add_executable( concurrentVX 
	main.cpp
  )
target_link_libraries(concurrentVX
  ${OpenCV_LIBS}
  ${VisionWorks_LIBRARIES}
  ${VisionWorks-NVXIO_LIBRARIES}
  opencv_dep_cudart
  )



