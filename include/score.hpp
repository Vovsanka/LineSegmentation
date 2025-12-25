#ifndef SCORE_HPP
#define SCORE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "color.hpp"


__host__ __device__
double getRad(int direction);

__host__ __device__
thrust::tuple<double,double> getUnitVector(int dir);

__host__ __device__ 
thrust::tuple<double,double> getOrthogonalUnitVector(int dir);

__host__ __device__ 
void insertionSort(double* a, int n);

__host__ __device__
double emd(const int* arr1, const int* arr2);

__host__ __device__
double computeLabScore(
    const uchar* F,
    size_t Fstep,
    double yPixel, double xPixel,
    int dir, 
    int width, int height
); 

__host__ __device__
thrust::tuple<double,int> bestPossibleScore(
    const uchar* F, size_t Fstep,
    double yPixel, double xPixel,
    int width, int height
);

#endif