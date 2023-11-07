cmake_minimum_required(VERSION 3.22)
project(nano_track_trt_text)

set(CMAKE_CXX_STANDARD 14)

# OpenCV
# change this DIR to your opencv dir
set(OpenCV_DIR /home/lizhang/opencv-4.x/build)
find_package(OpenCV)

# TensorRT
# 1.change this dir to your tensorrt dir
set(TensorRT_INCLUDE_DIRS /home/lizhang/TensorRT-8.5.2.2/include)
set(TensorRT_LIBRARIES /home/lizhang/TensorRT-8.5.2.2/lib)

# 2.if you use jetson with tensorrt , include your tensorrt include and lib like this:
# include_directories(/usr/include/aarch64-linux-gnu/)
# link_directories(/usr/lib/aarch64-linux-gnu/)

#CUDA
find_package(CUDA REQUIRED)
link_directories(/usr/local/cuda/lib64)

include_directories(
        ${TensorRT_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include/common.h
        ${CMAKE_SOURCE_DIR}/include/UltraFace.h
)

add_executable(Ultra_Face_trt main.cpp)

target_link_libraries(Ultra_Face_trt
        ${OpenCV_LIBS}
        libcudart.so
        libnvinfer.so
        libnvinfer_plugin.so
        ${TensorRT_LIBRARIES}
        ${CUDA_LIBRARIES}
        )
