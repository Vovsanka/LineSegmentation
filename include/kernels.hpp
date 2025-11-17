#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <iostream>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <cuda_runtime.h>
#include <thrust/pair.h>


void showMatrix(const cv::Mat &F);

void showImage(const cv::Mat &F);

__host__ __device__
thrust::pair<double,double> directionNormalUnitVector(int d);

__host__ __device__
double computeScore(const uchar* F,
                    double yPixel, double xPixel,
                    double unitNormY, double unitNormX,
                    int width, int height);

__global__
void bestScoreKernel(const uchar* F, double* S, int* D,
                     int width, int height);

__global__
void candidateThresholdKernel(const double *S, const uchar *D, uchar *C,
                              int width, int height);

__host__
std::pair<double,double> findFractionalCandidate(const cv::Mat &F,
                                                 int y, int x, int d);

__host__
std::pair<cv::Mat, cv::Mat> candidateIterativeSearch(const cv::Mat &F,
                                                     const cv::Mat &S,
                                                     const cv::Mat &D);

__host__ 
void rgbRun();

#endif