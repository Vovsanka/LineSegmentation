#ifndef SCORE_HPP
#define SCORE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "color.hpp"


__host__ __device__
float getRad(int direction);

__host__ __device__
thrust::tuple<float,float> getUnitVector(int dir);

__host__ __device__ 
thrust::tuple<float,float> getOrthogonalUnitVector(int dir);

__host__ __device__ 
void insertionSort(float* a, int n);

__host__ __device__
float emd(const int* arr1, const int* arr2);

__host__ __device__
float computeLabScore(
    const uchar* F,
    size_t Fstep,
    float yPixel, float xPixel,
    int dir, 
    int width, int height
); 

__host__ __device__
thrust::tuple<float,int> bestPossibleScore(
    const uchar* F, size_t Fstep,
    float yPixel, float xPixel,
    int width, int height
);

#endif