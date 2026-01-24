#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
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

__host__
void showScoreDirectionMatrix(
    cv::Mat &S,
    cv::Mat &D,
    std::vector<Cand>& candidates
);

__host__
void drawLines(const std::vector<Line>& lines, int width, int height);

#endif