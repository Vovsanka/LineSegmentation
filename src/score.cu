#include "score.hpp"

__host__ __device__
float getRad(int direction) {
    direction %= DIRECTIONS;
    return direction*(PI / DIRECTIONS);
}

__host__ __device__ 
thrust::tuple<float,float> getUnitVector(float rad) { // y x
    while (rad >= PI) rad -= PI;
    return thrust::make_tuple(sin(rad), cos(rad));
}

__host__ __device__ 
thrust::tuple<float,float> getOrthogonalUnitVector(float rad) { // y x
    rad += PI/2;
    return getUnitVector(rad);
}

__host__ __device__ 
void insertionSort(float* a, int n) {
    for (int i = 1; i < n; ++i) {
        float key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

__host__ __device__
float emd(const float* arr, int dir) {
    //
    int edge = (dir + DIRECTIONS/2) % DIRECTIONS;
    // array sum (in order to normalize)
    float sum = 0;
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        sum += arr[k];
    }
    // prefix sum (normalized values)
    float fixed = (1.0/DIRECTIONS); 
    float prefixSum1[2*DIRECTIONS], prefixSum2[2*DIRECTIONS];
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        float val1, val2;
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
        float arrVal = arr[k]/sum;
        float delta1 = arrVal - val1;
        float delta2 = arrVal - val2;
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
    float med1 = (prefixSum1[DIRECTIONS - 1] + prefixSum1[DIRECTIONS])/2;
    float med2 = (prefixSum2[DIRECTIONS - 1] + prefixSum2[DIRECTIONS])/2;
    // ring EMD
    float emd1 = 0.0f, emd2 = 0.0f;
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        emd1 += fabsf(prefixSum1[k] - med1);
        emd2 += fabsf(prefixSum2[k] - med2);
    }
    //
    return min(emd1, emd2); // the value is bounded by DIRECTIONS/2.0 (uniform <-> one side uniform)
}

__host__ __device__
float computeLabScore(
    const uchar* F,
    size_t Fstep,
    float yPixel, float xPixel,
    int dir, 
    int width, int height
) {
    float emdSum = 0.0f;
    for (int c = 1; c <= CIRCLE_COUNT; c++) {
        //
        int minL = 255, minA = 255, minB = 255;
        for (int d = 0; d < DIRECTIONS; d++) { 
            // 
            thrust::tuple<float,float> unit = getUnitVector(getRad(d));
            float dY = thrust::get<0>(unit);
            float dX = thrust::get<1>(unit);
            //
            float y1 = yPixel + c*dY;
            float x1 = xPixel + c*dX;
            //
            float y2 = yPixel - c*dY;
            float x2 = xPixel - c*dX;
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
        float lArr[2*DIRECTIONS];
        float aArr[2*DIRECTIONS];
        float bArr[2*DIRECTIONS];
        for (int d = 0; d < DIRECTIONS; d++) { 
            // 
            thrust::tuple<float,float> unit = getOrthogonalUnitVector(getRad(d));
            float dY = thrust::get<0>(unit);
            float dX = thrust::get<1>(unit);
            //
            float y1 = yPixel + c*dY;
            float x1 = xPixel + c*dX;
            //
            float y2 = yPixel - c*dY;
            float x2 = xPixel - c*dX;
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
    float maxEmd = 1.0f*DIRECTIONS/4;
    float emdAv = emdSum/CIRCLE_COUNT;
    float emdNorm = fminf(emdAv/maxEmd, 1.0f);
    float emdScore = 1.0 - powf(emdNorm, SCORE_BOOSTER); 
    return emdScore;
}

__host__ __device__
thrust::tuple<float,int> bestPossibleScore(
    const uchar* F, size_t Fstep,
    float yPixel, float xPixel,
    int width, int height
) {
    float bestScore = -1.0f;
    int bestDir = 0;
    for (int d = 0; d < DIRECTIONS; d++) {
        float score = computeLabScore(F, Fstep, yPixel, xPixel, d, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }
    // std::cout << "Best direction: " << bestDir << std::endl;
    // std::cout << "Best score: " << bestScore << std::endl;
    return thrust::make_tuple(bestScore, bestDir);
}
