# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/dev/vxProject/videoVX

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/dev/vxProject/videoVX

# Include any dependencies generated for this target.
include CMakeFiles/videoVX.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/videoVX.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/videoVX.dir/flags.make

CMakeFiles/videoVX.dir/main.cpp.o: CMakeFiles/videoVX.dir/flags.make
CMakeFiles/videoVX.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/dev/vxProject/videoVX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/videoVX.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/videoVX.dir/main.cpp.o -c /home/ubuntu/dev/vxProject/videoVX/main.cpp

CMakeFiles/videoVX.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/videoVX.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/dev/vxProject/videoVX/main.cpp > CMakeFiles/videoVX.dir/main.cpp.i

CMakeFiles/videoVX.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/videoVX.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/dev/vxProject/videoVX/main.cpp -o CMakeFiles/videoVX.dir/main.cpp.s

CMakeFiles/videoVX.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/videoVX.dir/main.cpp.o.requires

CMakeFiles/videoVX.dir/main.cpp.o.provides: CMakeFiles/videoVX.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/videoVX.dir/build.make CMakeFiles/videoVX.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/videoVX.dir/main.cpp.o.provides

CMakeFiles/videoVX.dir/main.cpp.o.provides.build: CMakeFiles/videoVX.dir/main.cpp.o


# Object files for target videoVX
videoVX_OBJECTS = \
"CMakeFiles/videoVX.dir/main.cpp.o"

# External object files for target videoVX
videoVX_EXTERNAL_OBJECTS =

videoVX: CMakeFiles/videoVX.dir/main.cpp.o
videoVX: CMakeFiles/videoVX.dir/build.make
videoVX: /usr/lib/libopencv_gpu.so.2.4.13
videoVX: /usr/lib/libvisionworks.so.1.5.3
videoVX: /usr/lib/libnvxio.so.1.5.3
videoVX: /usr/local/cuda-8.0/lib64/libcudart.so
videoVX: /usr/local/cuda-8.0/lib64/libcufft.so
videoVX: /usr/lib/libopencv_legacy.so.2.4.13
videoVX: /usr/lib/libopencv_video.so.2.4.13
videoVX: /usr/lib/libopencv_calib3d.so.2.4.13
videoVX: /usr/lib/libopencv_features2d.so.2.4.13
videoVX: /usr/lib/libopencv_flann.so.2.4.13
videoVX: /usr/lib/libopencv_ml.so.2.4.13
videoVX: /usr/lib/libopencv_objdetect.so.2.4.13
videoVX: /usr/lib/libopencv_highgui.so.2.4.13
videoVX: /usr/lib/libopencv_photo.so.2.4.13
videoVX: /usr/lib/libopencv_imgproc.so.2.4.13
videoVX: /usr/lib/libopencv_core.so.2.4.13
videoVX: /usr/local/cuda-8.0/lib64/libcudart.so
videoVX: /usr/local/cuda-8.0/lib64/libnppc.so
videoVX: /usr/local/cuda-8.0/lib64/libnppi.so
videoVX: /usr/local/cuda-8.0/lib64/libnpps.so
videoVX: CMakeFiles/videoVX.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/dev/vxProject/videoVX/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable videoVX"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/videoVX.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/videoVX.dir/build: videoVX

.PHONY : CMakeFiles/videoVX.dir/build

CMakeFiles/videoVX.dir/requires: CMakeFiles/videoVX.dir/main.cpp.o.requires

.PHONY : CMakeFiles/videoVX.dir/requires

CMakeFiles/videoVX.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/videoVX.dir/cmake_clean.cmake
.PHONY : CMakeFiles/videoVX.dir/clean

CMakeFiles/videoVX.dir/depend:
	cd /home/ubuntu/dev/vxProject/videoVX && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/dev/vxProject/videoVX /home/ubuntu/dev/vxProject/videoVX /home/ubuntu/dev/vxProject/videoVX /home/ubuntu/dev/vxProject/videoVX /home/ubuntu/dev/vxProject/videoVX/CMakeFiles/videoVX.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/videoVX.dir/depend

