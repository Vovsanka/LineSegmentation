#include "score.hpp"

__host__ __device__
double getRad(int d) { // returns [0, 2*PI)
    d %= 2*DIRECTIONS;
    return d*PI/DIRECTIONS;
}

__host__ __device__ 
thrust::tuple<double,double> getUnitVector(int d) { // (y, x) // d in [0, 2*DIRECTIONS)
    double rad = getRad(d);
    return thrust::make_tuple(sin(rad), cos(rad));
}

__host__ __device__
int getOrthogonalDirection(int d) { // d in [0, 2*DIRECTIONS)
    return (d + DIRECTIONS/2) % (2*DIRECTIONS);
}

__host__ __device__
int getOppositeDirection(int d) { // d in [0, 2*DIRECTIONS)
    return (d + DIRECTIONS) % (2*DIRECTIONS);
}

__host__ __device__ 
thrust::tuple<double,double> getOrthogonalUnitVector(int d) { // (y, x) // d in [0, 2*DIRECTIONS)
    return getUnitVector(getOrthogonalDirection(d));
}

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
double emd(const double* arr, int d) { // d in [0, 2*DIRECTIONS)
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
    int d, int c,
    int width, int height
) { // d in [0, 2*DIRECTIONS)
    thrust::tuple<double,double> unitVector = getOrthogonalUnitVector(d);
    double dy = thrust::get<0>(unitVector);
    double dx = thrust::get<1>(unitVector);
    //
    double yShifted = y + c*dy;
    double xShifted = x + c*dx;
    //
    return getColorChannels(F, Fstep, yShifted, xShifted, width, height);
}

__host__ __device__
double computeLabScore(
    const uchar* F,
    size_t Fstep,
    double yPixel, double xPixel,
    int dir, 
    int width, int height
) { // dir in [0, DIRECTIONS)
    double emdSum = 0.0;
    for (int c = 1; c <= CIRCLE_COUNT; c++) {
        //
        int minL = 255, minA = 255, minB = 255;
        double lArr[2*DIRECTIONS];
        double aArr[2*DIRECTIONS];
        double bArr[2*DIRECTIONS];
        for (int d1 = 0; d1 < DIRECTIONS; d1++) { 
            int d2 = getOppositeDirection(d1); 
            //
            thrust::tuple<uchar,uchar,uchar> lab1 = getShiftedColorChannels(
                F, Fstep, yPixel, xPixel, d1, c, width, height);
            int l1 = thrust::get<0>(lab1);
            int a1 = thrust::get<1>(lab1);
            int b1 = thrust::get<2>(lab1);
            //
            thrust::tuple<uchar,uchar,uchar> lab2 = getShiftedColorChannels(
                F, Fstep, yPixel, xPixel, d2, c, width, height);
            int l2 = thrust::get<0>(lab2);
            int a2 = thrust::get<1>(lab2);
            int b2 = thrust::get<2>(lab2);
            //
            minL = min(minL, min(l1, l2));
            minA = min(minA, min(a1, a2));
            minB = min(minB, min(b1, b2));
            // 
            lArr[d1] = l1;
            aArr[d1] = a1;
            bArr[d1] = b1;
            //
            lArr[d2] = l2;
            aArr[d2] = a2;
            bArr[d2] = b2;
          }
        //
        for (int d1 = 0; d1 < DIRECTIONS; d1++) { 
            int d2 = getOppositeDirection(d1); 
            //
            lArr[d1] += COLOR_OFFSET - minL;
            aArr[d1] += COLOR_OFFSET - minA;
            bArr[d1] += COLOR_OFFSET - minB;
            //
            lArr[d2] += COLOR_OFFSET - minL;
            aArr[d2] += COLOR_OFFSET - minA;
            bArr[d2] += COLOR_OFFSET - minB;
        }
        // 
        emdSum += min(emd(lArr, dir), min(emd(aArr, dir), emd(bArr, dir)));
    }
    // compute the EMD-score
    double maxEmd = 1.0*DIRECTIONS/4;
    double emdAv = emdSum/CIRCLE_COUNT;
    double emdNorm = fmin(emdAv/maxEmd, 1.0);
    double emdScore = 1.0 - pow(emdNorm, SCORE_BOOSTER); 
    return emdScore;
}

__host__ __device__
thrust::tuple<double,int> bestPossibleScore(
    const uchar* F, size_t Fstep,
    double yPixel, double xPixel,
    int width, int height
) {
    double bestScore = -1.0;
    int bestDir = 0;
    for (int d = 0; d < DIRECTIONS; d++) {
        double score = computeLabScore(F, Fstep, yPixel, xPixel, d, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }
    // std::cout << "Best direction: " << bestDir << std::endl;
    // std::cout << "Best score: " << bestScore << std::endl;
    return thrust::make_tuple(bestScore, bestDir);
}
