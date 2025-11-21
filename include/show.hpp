#ifndef SHOW_HPP
#define SHOW_HPP

#include <opencv2/opencv.hpp>

__host__
cv::Mat downloadToCpu(const cv::cuda::GpuMat& gpuF);

__host__
void showMatrix(const cv::cuda::GpuMat& gpuF);

__host__
void showImage(const cv::cuda::GpuMat& gpuF);


#endif