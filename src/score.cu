#include "score.hpp"


__host__ __device__ 
void insertionSort(double* a, int n) {
    for (int i = 1; i < n; ++i) {
        double key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

__host__ __device__
double emdRing(const double* arr, int d) { // d in [0, 2*DIRECTIONS)
    int edge1 = getOrthogonalDirection(d);
    int edge2 = getOppositeDirection(edge1);
    if (edge1 > edge2) { // make edge1 < edge2
        int temp = edge1;
        edge1 = edge2;
        edge2 = temp;
    }
    // array sum (in order to normalize)
    double sum = 0;
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        sum += arr[k];
    }
    // prefix sum (normalized values)
    double fixed = (1.0/DIRECTIONS); 
    double prefixSum1[2*DIRECTIONS], prefixSum2[2*DIRECTIONS];
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        double val1, val2;
        if (k == edge1 || k == edge2) {
            val1 = val2 = fixed/2.0; 
        } else if (k < edge1 || edge2 < k) {
            val1 = 0.0;
            val2 = fixed;
        } else { 
            val1 = fixed;
            val2 = 0.0;
        }
        // prefix sums of delta of the normalized array values 
        double arrVal = arr[k]/sum;
        double delta1 = arrVal - val1;
        double delta2 = arrVal - val2;
        if (!k) {
            prefixSum1[k] = delta1;
            prefixSum2[k] = delta2;
        } else {
            prefixSum1[k] = prefixSum1[k - 1] + delta1;
            prefixSum2[k] = prefixSum2[k - 1] + delta2;
        }
    }
    // median computation
    insertionSort(prefixSum1, 2*DIRECTIONS);
    insertionSort(prefixSum2, 2*DIRECTIONS);
    double med1 = (prefixSum1[DIRECTIONS - 1] + prefixSum1[DIRECTIONS])/2.0;
    double med2 = (prefixSum2[DIRECTIONS - 1] + prefixSum2[DIRECTIONS])/2.0; // ring EMD
    double emd1 = 0.0, emd2 = 0.0;
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        emd1 += fabs(prefixSum1[k] - med1);
        emd2 += fabs(prefixSum2[k] - med2);
    }
    //
    return fmin(emd1, emd2); // the value is bounded by DIRECTIONS/2.0 (uniform <-> one side uniform)
}

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getShiftedColorChannels(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int d,
    int width, int height
) { // d in [0, 2*DIRECTIONS)
    Vec unitVector = getUnitVector(d);
    //
    double yShifted = y + CIRCLE_RADIUS*unitVector.y;
    double xShifted = x + CIRCLE_RADIUS*unitVector.x;
    //
    return getColorChannels(F, Fstep, yShifted, xShifted, width, height);
}


__host__ __device__
Cand bestPossibleScoreDirection(
    const uchar* F, size_t Fstep,
    double yPixel, double xPixel,
    int width, int height
) {
    double bestScore = computeLabScore(F, Fstep, yPixel, xPixel, 0, width, height);
    int bestDir = 0;
    for (int d = 1; d < DIRECTIONS; d++) {
        double score = computeLabScore(F, Fstep, yPixel, xPixel, d, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }
    return Cand(yPixel, xPixel, bestDir, bestScore);
}
