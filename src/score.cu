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
float emd(const int* arr1, const int* arr2) {
    // normalize the arrays
    int sum1 = 0, sum2 = 0;
    for (int k = 0; k < DIRECTIONS; k++) {
        sum1 += arr1[k];
        sum2 += arr2[k];
    }
    // 
    float cum1 = 0.0f, cum2 = 0.0f;
    float emd = 0.0f;
    for (int k = 0; k < DIRECTIONS; k++) {
        if (sum1 > 0) cum1 += 1.0f*arr1[k]/sum1;
        if (sum2 > 0) cum2 += 1.0f*arr2[k]/sum2;
        emd += fabsf(cum1 - cum2);
    }
    return emd;
}

__host__ __device__
float computeLabScore(
    const uchar* F,
    float yPixel, float xPixel,
    int dir, 
    int width, int height
) {
    for (int c = 1; c <= CIRCLE_COUNT; c++) {
        int minL = 255, minA = 255, minB = 255;
        int lArr1[DIRECTIONS - 1], lArr2[DIRECTIONS - 1];
        int aArr1[DIRECTIONS - 1], aArr2[DIRECTIONS - 1];
        int bArr1[DIRECTIONS - 1], bArr2[DIRECTIONS - 1];
        for (int k = 1; k < DIRECTIONS; k++) { 
            // define the current direction (skip the line direction)
            int d = (dir + k) % DIRECTIONS;
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
        for (int k = 1; k < DIRECTIONS; k++) { 
            // define the current direction (skip the line direction)
            int d = (dir + k) % DIRECTIONS;
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
            int ind = k - 1;
            lArr1[ind] = l1;
            lArr2[ind] = l2;
            aArr1[ind] = a1;
            aArr2[ind] = a2;
            bArr1[ind] = b1;
            bArr2[ind] = b2;
        }
        emd(lArr1, lArr2);
    }
    // compute the score
    return 0.0f; 
    // float lRatio = max(l1/l2, l2/l1);
    // float abRatio = max(ab1/ab2, ab2/ab1);
    // float ratio = max(lRatio, abRatio);
    // return 1.0 - std::pow(1.0/ratio, SCORE_EXP);
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
