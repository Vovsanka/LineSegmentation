#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include <string>

#include <cairo/cairo.h>

#include "config.hpp"
#include "cand_type.hpp"
#include "directions.hpp"
#include "line_type.hpp"


template <typename T>
__host__ __device__
T& cell(
    const T* Fptr, size_t Fstep,
    int y, int x
) { 
    T* rowF = (T*)((uchar*)Fptr + y * Fstep);
    return rowF[x];
}

__host__
cv::cuda::GpuMat uploadToGPU(const cv::Mat& cpuF);

__host__
cv::Mat downloadToCPU(const cv::cuda::GpuMat& gpuF);

__host__
void showImage(std::string name, const cv::Mat& cpuF);

__host__
void showImage(std::string name, const cv::cuda::GpuMat& gpuF);

__host__
void showMatrix(std::string name, const cv::Mat &cpuF);

__host__
void showMatrix(std::string name, const cv::cuda::GpuMat& gpuF);

#endif