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
float computeLabScore(
    const uchar* F,
    float yPixel, float xPixel,
    int dir, 
    int width, int height
) {
    // TODO
    for (int c = 1; c <= CIRCLE_COUNT; c++) {
        for (int d = 0; d < DIRECTIONS; d++) {
            // skip the line direction
            if (d == dir) continue; 
            // 
            thrust::tuple<float,float> unit = getOrthogonalUnitVector(getRad(dir));
            float dY = thrust::get<0>(unit);
            float dX = thrust::get<1>(unit);
            //
            float y1 = yPixel + c*dY;
            float x1 = xPixel + c*dX;
            //
            float y2 = yPixel - c*dY;
            float x2 = xPixel - c*dX;
            //
            // TODO: get the colors add add them to the EMD distributions 
        }
    }
    // float l1 = 0, l2 = 0, ab1 = 0, ab2 = 0;
    // int minL = 255, minA = 255, minB = 255;
    // //
    // thrust::tuple<float,float> unitNorm = getUnitVector(dirRad);
    // float unitNormY = thrust::get<0>(unitNorm);
    // float unitNormX = thrust::get<1>(unitNorm); 
    // //
    // float y1 = yPixel - R, y2 = yPixel + R;
    // float x1 = xPixel - R, x2 = xPixel + R;
    // for (float y = y1; y <= y2; y += STEP) {
    //     for (float x = x1; x <= x2; x += STEP) {
    //         // check if in the or on the circle
    //         float dy = (y - yPixel), dx = (x - xPixel);
    //         if (dy*dy + dx*dx > R*R) continue;
    //         //
    //         float signedDist = dy*unitNormY + dx*unitNormX;
    //         float dist = abs(signedDist);
    //         // skip the pixels on the line
    //         if (dist < THICKNESS/2) continue;
    //         //
    //         thrust::tuple<uchar,uchar,uchar> lab = getColorChannels(F, y, x, width, height);
    //         int l = thrust::get<0>(lab);
    //         int a = thrust::get<1>(lab);
    //         int b = thrust::get<2>(lab);
    //         minL = min(minL, l);
    //         minA = min(minA, a);
    //         minB = min(minB, b);
    //     }
    // }
    // for (float y = y1; y <= y2; y += STEP) {
    //     for (float x = x1; x <= x2; x += STEP) {
    //         // check if in the or on the circle
    //         float dy = (y - yPixel), dx = (x - xPixel);
    //         if (dy*dy + dx*dx > R*R) continue;
    //         //
    //         float signedDist = dy*unitNormY + dx*unitNormX;
    //         float dist = abs(signedDist);
    //         // skip the pixels on the line
    //         if (dist < THICKNESS/2) continue;
    //         //
    //         float w =  1.0 - sqrt(dx*dx + dy*dy)/R;
    //         //
    //         thrust::tuple<uchar,uchar,uchar> lab = getColorChannels(F, y, x, width, height);
    //         int l = thrust::get<0>(lab);
    //         int a = thrust::get<1>(lab);
    //         int b = thrust::get<2>(lab);
    //         //
    //         float lContribution = w*(l - minL + OFFSET);
    //         float abContribution = w*((a - minA + OFFSET) + (b - minB + OFFSET));
    //         // equalize intensive and non-intensive colors
    //         if (signedDist > 0) { // the half-circle of the normal vector
    //             l1 += lContribution;
    //             ab1 += abContribution;
    //         } else { // the half-circle opposite to the normal vector
    //             l2 += lContribution;
    //             ab2 += abContribution;
    //         }
    //     }
    // }
    // // compute the score
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
