#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "config.hpp"


struct Vec {
    double y, x;    

    Vec() = default;

    __host__ __device__
    Vec(double y, double x);
};

__host__
dim3 getGrid(int width, int height);

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
cv::Mat convertBGRtoLab(const cv::Mat& cpuF);

__host__
cv::Mat filterNoise(const cv::Mat& cpuF);

__host__
double computeScale(const cv::Mat& cpuF);

__host__
cv::Mat resize(const cv::Mat& cpuF, double scale);

__host__
void showImage(const cv::Mat& cpuF);

__host__
void showImage(const cv::cuda::GpuMat& gpuF);

__host__
void showMatrix(const cv::Mat &cpuF);

__host__
void showMatrix(const cv::cuda::GpuMat& gpuF);

#endif