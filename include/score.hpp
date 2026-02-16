#ifndef SCORE_HPP
#define SCORE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/tuple.h>

#include "config.hpp"
#include "color.hpp"
#include "directions.hpp"


__host__ __device__ 
void insertionSort(double* a, int n);

__host__ __device__
double emdRing(const double* arr, int d);

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getShiftedColorChannels(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int d,
    int width, int height
); // d in [0, 2*DIRECTIONS)

__host__ __device__
inline double computeLabScore(
    const uchar* F,
    size_t Fstep,
    double yPixel, double xPixel,
    int dir, 
    int width, int height
) {
    int minL = 255, minA = 255, minB = 255;
    double lArr[2*DIRECTIONS];
    double aArr[2*DIRECTIONS];
    double bArr[2*DIRECTIONS];

    for (int k = 0; k < 2*DIRECTIONS; ++k) {
        lArr[k] = 0.0;
        aArr[k] = 0.0;
        bArr[k] = 0.0;
    }

    for (int d1 = 0; d1 < DIRECTIONS; d1++) {
        int d2 = getOppositeDirection(d1);

        auto lab1 = getShiftedColorChannels(F, Fstep, yPixel, xPixel, d1, width, height);
        int l1 = thrust::get<0>(lab1);
        int a1 = thrust::get<1>(lab1);
        int b1 = thrust::get<2>(lab1);

        auto lab2 = getShiftedColorChannels(F, Fstep, yPixel, xPixel, d2, width, height);
        int l2 = thrust::get<0>(lab2);
        int a2 = thrust::get<1>(lab2);
        int b2 = thrust::get<2>(lab2);

        minL = (minL < l1 ? minL : l1);
        minL = (minL < l2 ? minL : l2);

        minA = (minA < a1 ? minA : a1);
        minA = (minA < a2 ? minA : a2);

        minB = (minB < b1 ? minB : b1);
        minB = (minB < b2 ? minB : b2);

        lArr[d1] = l1; aArr[d1] = a1; bArr[d1] = b1;
        lArr[d2] = l2; aArr[d2] = a2; bArr[d2] = b2;
    }

    for (int d1 = 0; d1 < DIRECTIONS; d1++) {
        int d2 = getOppositeDirection(d1);

        lArr[d1] += COLOR_OFFSET - minL;
        aArr[d1] += COLOR_OFFSET - minA;
        bArr[d1] += COLOR_OFFSET - minB;

        lArr[d2] += COLOR_OFFSET - minL;
        aArr[d2] += COLOR_OFFSET - minA;
        bArr[d2] += COLOR_OFFSET - minB;
    }

    double emdCircle = fmin(
        emdRing(lArr, dir),
        fmin(emdRing(aArr, dir), emdRing(bArr, dir))
    );
    double emdMax  = 1.0 * DIRECTIONS / 4.0;
    double emdBest = fmin(emdMax, emdCircle);
    double emdScore = 1.0 - emdBest / emdMax;

    return emdScore;
};

__host__ __device__
Cand bestPossibleScoreDirection(
    const uchar* F, size_t Fstep,
    double yPixel, double xPixel,
    int width, int height
);

#endif