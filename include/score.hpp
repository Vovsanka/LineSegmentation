#ifndef SCORE_HPP
#define SCORE_HPP

#include "config.hpp"
#include "color.hpp"
#include "directions.hpp"


__host__ __device__
thrust::tuple<double,double,double> computeStructureTensor( // Jxx, Jyy, Jxy
    const uchar* F,
    size_t Fstep,
    double yPixel, double xPixel,
    int width, int height
);

__host__ __device__
void shellSort(double* a, int n);

__host__ __device__
double emdRing(const double* arr, int d);

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getShiftedColorChannels(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int d, int c,
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
    double emdCircle[CIRCLE_COUNT + 1];
    //
    for (int c = 1; c <= CIRCLE_COUNT; c++) {
        //
        int minL = 255, minA = 255, minB = 255;
        //
        double lArr[2*DIRECTIONS];
        double aArr[2*DIRECTIONS];
        double bArr[2*DIRECTIONS];
        //
        for (int d1 = 0; d1 < DIRECTIONS; d1++) {
            int d2 = getOppositeDirection(d1);

            auto lab1 = getShiftedColorChannels(F, Fstep, yPixel, xPixel, d1, c, width, height);
            int l1 = thrust::get<0>(lab1);
            int a1 = thrust::get<1>(lab1);
            int b1 = thrust::get<2>(lab1);

            auto lab2 = getShiftedColorChannels(F, Fstep, yPixel, xPixel, d2, c, width, height);
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
        emdCircle[c] = fmin(
            emdRing(lArr, dir),
            fmin(emdRing(aArr, dir), emdRing(bArr, dir))
        );
    }
    //
    double emdMax  = 1.0*DIRECTIONS/4.0;
    double weightedSum = 0.0, weight = 0.0;
    for (int c = 1; c <= CIRCLE_COUNT; c++) {
        double circleScore = 1.0 - fmin(emdMax, emdCircle[c]) / emdMax;
        // int w = (CIRCLE_COUNT - c + 1);
        int w = 1;
        weightedSum += circleScore*w;
        weight += w;
    }
    double score = weightedSum / weight;
    //
    return score;
};

__host__ __device__
inline double computeGrayScore(
    const uchar* F,
    size_t Fstep,
    double yPixel, double xPixel,
    int dir, 
    int width, int height
) {
    thrust::tuple<double,double,double> structureTensor = computeStructureTensor(
        F, Fstep, yPixel, xPixel, width, height
    );
    double Jxx = thrust::get<0>(structureTensor);
    double Jyy = thrust::get<1>(structureTensor);
    double Jxy = thrust::get<2>(structureTensor);
    // 
    Vec normalUnitVector = getUnitVector(dir);
    double nx = normalUnitVector.x;
    double ny = normalUnitVector.y;
    // tangent (orthogonal to normal unit vector)  
    double tx = -ny;
    double ty = nx;
    //
    double En = nx*nx*Jxx + 2.0*nx*ny*Jxy + ny*ny*Jyy;
    double Et = tx*tx*Jxx + 2.0*tx*ty*Jxy + ty*ty*Jyy;
    // 1. Directional anisotropy in [0,1] 
    double rawScoreDir = (En - Et) / (En + Et + 1e-12); 
    double scoreDir = 0.5 * (rawScoreDir + 1.0); 
    // 2. Edge strength in [0,1] 
    double scoreEdge = En / (En + EDGE_SHARPNESS); 
    // 3. Combined score 
    return scoreDir*scoreEdge;
}

__host__ __device__
Cand bestPossibleScoreDirection(
    const uchar* F, size_t Fstep,
    double yPixel, double xPixel,
    int width, int height,
    bool beamScore = true
);

#endif