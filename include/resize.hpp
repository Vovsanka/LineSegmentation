#ifndef RESIZE_HPP
#define RESIZE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>

#include "config.hpp"

cv::cuda::GpuMat resize(const cv::cuda::GpuMat& F);


#endif