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
CMAKE_SOURCE_DIR = /home/ubuntu/dev/vxProject/cannyCV_CPUGPU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/dev/vxProject/cannyCV_CPUGPU

# Include any dependencies generated for this target.
include CMakeFiles/cannyCV_CPUGPU.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cannyCV_CPUGPU.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cannyCV_CPUGPU.dir/flags.make

CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o: CMakeFiles/cannyCV_CPUGPU.dir/flags.make
CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/dev/vxProject/cannyCV_CPUGPU/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o -c /home/ubuntu/dev/vxProject/cannyCV_CPUGPU/main.cpp

CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/dev/vxProject/cannyCV_CPUGPU/main.cpp > CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.i

CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/dev/vxProject/cannyCV_CPUGPU/main.cpp -o CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.s

CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o.requires

CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o.provides: CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/cannyCV_CPUGPU.dir/build.make CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o.provides

CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o.provides.build: CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o


# Object files for target cannyCV_CPUGPU
cannyCV_CPUGPU_OBJECTS = \
"CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o"

# External object files for target cannyCV_CPUGPU
cannyCV_CPUGPU_EXTERNAL_OBJECTS =

cannyCV_CPUGPU: CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o
cannyCV_CPUGPU: CMakeFiles/cannyCV_CPUGPU.dir/build.make
cannyCV_CPUGPU: /usr/lib/libopencv_gpu.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libvisionworks.so.1.5.3
cannyCV_CPUGPU: /usr/lib/libnvxio.so.1.5.3
cannyCV_CPUGPU: /usr/local/cuda-8.0/lib64/libcudart.so
cannyCV_CPUGPU: /usr/local/cuda-8.0/lib64/libcufft.so
cannyCV_CPUGPU: /usr/lib/libopencv_legacy.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_video.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_calib3d.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_features2d.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_flann.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_ml.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_objdetect.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_highgui.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_photo.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_imgproc.so.2.4.13
cannyCV_CPUGPU: /usr/lib/libopencv_core.so.2.4.13
cannyCV_CPUGPU: /usr/local/cuda-8.0/lib64/libcudart.so
cannyCV_CPUGPU: /usr/local/cuda-8.0/lib64/libnppc.so
cannyCV_CPUGPU: /usr/local/cuda-8.0/lib64/libnppi.so
cannyCV_CPUGPU: /usr/local/cuda-8.0/lib64/libnpps.so
cannyCV_CPUGPU: CMakeFiles/cannyCV_CPUGPU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/dev/vxProject/cannyCV_CPUGPU/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cannyCV_CPUGPU"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cannyCV_CPUGPU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cannyCV_CPUGPU.dir/build: cannyCV_CPUGPU

.PHONY : CMakeFiles/cannyCV_CPUGPU.dir/build

CMakeFiles/cannyCV_CPUGPU.dir/requires: CMakeFiles/cannyCV_CPUGPU.dir/main.cpp.o.requires

.PHONY : CMakeFiles/cannyCV_CPUGPU.dir/requires

CMakeFiles/cannyCV_CPUGPU.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cannyCV_CPUGPU.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cannyCV_CPUGPU.dir/clean

CMakeFiles/cannyCV_CPUGPU.dir/depend:
	cd /home/ubuntu/dev/vxProject/cannyCV_CPUGPU && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/dev/vxProject/cannyCV_CPUGPU /home/ubuntu/dev/vxProject/cannyCV_CPUGPU /home/ubuntu/dev/vxProject/cannyCV_CPUGPU /home/ubuntu/dev/vxProject/cannyCV_CPUGPU /home/ubuntu/dev/vxProject/cannyCV_CPUGPU/CMakeFiles/cannyCV_CPUGPU.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cannyCV_CPUGPU.dir/depend

