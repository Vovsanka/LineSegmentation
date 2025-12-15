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
void merge(float* arr, int l, int m, int r) {
    int p1 = l;
    int p2 = m + 1;
    int interval = r - l + 1;
    float temp[2*DIRECTIONS];
    for (int k = 0; k < interval; k++) {
        if (p1 > m) {
            temp[k] = arr[p2++];
        } else if (p2 > r) {
            temp[k] = arr[p1++];
        } else if (arr[p1] < arr[p2]) {
            temp[k] = arr[p1++];
        } else {
            temp[k] = arr[p2++];
        }
    }
    //
    for (int k = 0; k < interval; k++) {
        arr[l + k] = temp[k];
    }
}

__host__ __device__
void mergeSort(float *arr, int l = 0, int r = 2*DIRECTIONS - 1) {
    // indices [l, r]
    if (l < r) {
        int m = l + (r - l)/2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

__host__ /*__device__*/
float emd(const float* arr, int dir) {
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
        if (k % DIRECTIONS == dir) {
            val1 = val2 = fixed/2; 
        } else if (k < dir || k > DIRECTIONS + dir) {
            val1 = 0;
            val2 = fixed;
        } else { // dir < k && k < DIRECTIONS + dir
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
    mergeSort(prefixSum1);
    mergeSort(prefixSum2);
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

__host__ /*__device__*/
float computeLabScore(
    const uchar* F,
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
            thrust::tuple<uchar,uchar,uchar> lab1 = getColorChannels(F, y1, x1, width, height);
            int l1 = thrust::get<0>(lab1);
            int a1 = thrust::get<1>(lab1);
            int b1 = thrust::get<2>(lab1);
            //
            thrust::tuple<uchar,uchar,uchar> lab2 = getColorChannels(F, y2, x2, width, height);
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
            thrust::tuple<uchar,uchar,uchar> lab1 = getColorChannels(F, y1, x1, width, height);
            int l1 = thrust::get<0>(lab1);
            int a1 = thrust::get<1>(lab1);
            int b1 = thrust::get<2>(lab1);
            //
            thrust::tuple<uchar,uchar,uchar> lab2 = getColorChannels(F, y2, x2, width, height);
            int l2 = thrust::get<0>(lab2);
            int a2 = thrust::get<1>(lab2);
            int b2 = thrust::get<2>(lab2);
            // 
            lArr[d] = l1 - minL + 0.1; // -0.1 to fix the 0-arrays
            lArr[DIRECTIONS + d] = l2 - minL + 0.1;
            //
            aArr[d] = a1 - minA + 0.1;
            aArr[DIRECTIONS + d] = a2 - minA + 0.1;
            //
            bArr[d] = b1 - minB + 0.1;
            bArr[DIRECTIONS + d] = b2 - minB + 0.1;
        }
        //
        emdSum += min(emd(lArr, dir), min(emd(bArr, dir), emd(aArr, dir)));
    }
    // compute the EMD-score
    float maxEmd = 1.0f*DIRECTIONS/4;
    float emdAv = emdSum/CIRCLE_COUNT;
    float emdNorm = fminf(emdAv/maxEmd, 1.0f);
    float emdScore = 1.0 - std::pow(emdNorm, SCORE_BOOSTER); 
    return emdScore;
}

// __host__ /*__device__*/
// thrust::tuple<float,float> bestPossibleScore(const uchar* F,
//                                                float yPixel, float xPixel,
//                                                int width, int height) {
//     // n-search
//     float l = 0, r = getPi();
//     while (r - l > DIR_PRECISION) {
//         float m[N + 1];
//         m[0] = l;
//         m[N] = r;
//         // only for points in between
//         for (int j = 1; j < N - 1; j++) {
//             m[j] = l + (j + 1)*(r - l)/N;
//         }
//         // only for points in between
//         float s[N];
//         int bestJ = 1;
//         for (int j = 1; j < N - 1; j++) {
//             s[j] = computeLabScore(F, yPixel, xPixel, m[j], width, height);
//             if (s[j] > s[bestJ]) {
//                 bestJ = j;
//             }
//         }
//         // update the range of the directions
//         l = m[bestJ - 1];
//         r = m[bestJ + 1];
//     }
//     //
//     float bestDir = (l + r)/2;
//     float bestScore = computeLabScore(F, yPixel, xPixel, bestDir, width, height);
//     // std::cout << "Best direction: " << bestDir << std::endl;
//     // std::cout << "Best score: " << bestScore << std::endl;
//     return thrust::make_tuple(bestScore, bestDir);
// }
