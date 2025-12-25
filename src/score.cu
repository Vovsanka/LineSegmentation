#include "score.hpp"

__host__ __device__
double getRad(int direction) {
    direction %= DIRECTIONS;
    return direction*(PI / DIRECTIONS);
}

__host__ __device__ 
thrust::tuple<double,double> getUnitVector(int dir) { // y x
    double rad = getRad(dir);
    while (rad >= PI) rad -= PI;
    return thrust::make_tuple(sin(rad), cos(rad));
}

__host__ __device__ 
thrust::tuple<double,double> getOrthogonalUnitVector(int dir) { // y x
    dir = (dir + DIRECTIONS/2) % DIRECTIONS;
    return getUnitVector(dir);
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
double emd(const double* arr, int dir) {
    //
    int edge = (dir + DIRECTIONS/2) % DIRECTIONS;
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
        if (k % DIRECTIONS == edge) {
            val1 = val2 = fixed/2; 
        } else if (k < edge || k > DIRECTIONS + edge) {
            val1 = 0;
            val2 = fixed;
        } else { // edge < k && k < DIRECTIONS + edge
            val1 = fixed;
            val2 = 0;
        }
        // std::cout << val1 << " | " << val2 << std::endl;
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
    double med1 = (prefixSum1[DIRECTIONS - 1] + prefixSum1[DIRECTIONS])/2;
    double med2 = (prefixSum2[DIRECTIONS - 1] + prefixSum2[DIRECTIONS])/2;
    // ring EMD
    double emd1 = 0.0, emd2 = 0.0;
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        emd1 += fabsf(prefixSum1[k] - med1);
        emd2 += fabsf(prefixSum2[k] - med2);
    }
    //
    return min(emd1, emd2); // the value is bounded by DIRECTIONS/2.0 (uniform <-> one side uniform)
}

__host__ __device__
double computeLabScore(
    const uchar* F,
    size_t Fstep,
    double yPixel, double xPixel,
    int dir, 
    int width, int height
) {
    double emdSum = 0.0;
    for (int c = 1; c <= CIRCLE_COUNT; c++) {
        //
        int minL = 255, minA = 255, minB = 255;
        for (int d = 0; d < DIRECTIONS; d++) { 
            // 
            thrust::tuple<double,double> unit = getOrthogonalUnitVector(d);
            double dY = thrust::get<0>(unit);
            double dX = thrust::get<1>(unit);
            //
            double y1 = yPixel + c*dY;
            double x1 = xPixel + c*dX;
            //
            double y2 = yPixel - c*dY;
            double x2 = xPixel - c*dX;
            //
            thrust::tuple<uchar,uchar,uchar> lab1 = getColorChannels(F, Fstep, y1, x1, width, height);
            int l1 = thrust::get<0>(lab1);
            int a1 = thrust::get<1>(lab1);
            int b1 = thrust::get<2>(lab1);
            //
            thrust::tuple<uchar,uchar,uchar> lab2 = getColorChannels(F, Fstep, y2, x2, width, height);
            int l2 = thrust::get<0>(lab2);
            int a2 = thrust::get<1>(lab2);
            int b2 = thrust::get<2>(lab2);
            //
            minL = min(minL, min(l1, l2));
            minA = min(minA, min(a1, a2));
            minB = min(minB, min(b1, b2));
        }
        //
        double lArr[2*DIRECTIONS];
        double aArr[2*DIRECTIONS];
        double bArr[2*DIRECTIONS];
        for (int d = 0; d < DIRECTIONS; d++) { 
            // 
            thrust::tuple<double,double> unit = getOrthogonalUnitVector(d);
            double dY = thrust::get<0>(unit);
            double dX = thrust::get<1>(unit);
            //
            double y1 = yPixel + c*dY;
            double x1 = xPixel + c*dX;
            //
            double y2 = yPixel - c*dY;
            double x2 = xPixel - c*dX;
            //
            thrust::tuple<uchar,uchar,uchar> lab1 = getColorChannels(F, Fstep, y1, x1, width, height);
            int l1 = thrust::get<0>(lab1);
            int a1 = thrust::get<1>(lab1);
            int b1 = thrust::get<2>(lab1);
            //
            thrust::tuple<uchar,uchar,uchar> lab2 = getColorChannels(F, Fstep, y2, x2, width, height);
            int l2 = thrust::get<0>(lab2);
            int a2 = thrust::get<1>(lab2);
            int b2 = thrust::get<2>(lab2);
            // 
            lArr[d] = l1 - minL + COLOR_OFFSET; // -COLOR_OFFSET to fix the 0-arrays
            lArr[DIRECTIONS + d] = l2 - minL + COLOR_OFFSET;
            //
            aArr[d] = a1 - minA + COLOR_OFFSET;
            aArr[DIRECTIONS + d] = a2 - minA + COLOR_OFFSET;
            //
            bArr[d] = b1 - minB + COLOR_OFFSET;
            bArr[DIRECTIONS + d] = b2 - minB + COLOR_OFFSET;
        }
        //
        emdSum += min(emd(lArr, dir), min(emd(bArr, dir), emd(aArr, dir)));
    }
    // compute the EMD-score
    double maxEmd = 1.0*DIRECTIONS/4;
    double emdAv = emdSum/CIRCLE_COUNT;
    double emdNorm = fminf(emdAv/maxEmd, 1.0);
    double emdScore = 1.0 - powf(emdNorm, SCORE_BOOSTER); 
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
