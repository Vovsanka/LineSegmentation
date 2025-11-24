#include "score.hpp"

__host__ __device__ 
inline thrust::tuple<double,double> getUnitVector(double rad) {
    return thrust::make_tuple(sin(rad), cos(rad));
}


__host__ __device__
double getRad(int direction) {
    const double PI = acos(-1.0);
    return direction*(PI / DIRECTIONS);
}

__host__ __device__
double computeLabScore(const uchar* F,
                       double yPixel, double xPixel,
                       double dirRad, 
                       int width, int height) {
    double l1 = 0, l2 = 0, ab1 = 0, ab2 = 0;
    int minL = 255, minA = 255, minB = 255;
    //
    thrust::tuple<double,double> unitNorm = getUnitVector(dirRad);
    double unitNormY = thrust::get<0>(unitNorm);
    double unitNormX = thrust::get<1>(unitNorm); 
    //
    double y1 = yPixel - R, y2 = yPixel + R;
    double x1 = xPixel - R, x2 = xPixel + R;
    for (double y = y1; y <= y2; y += STEP) {
        for (double x = x1; x <= x2; x += STEP) {
            // check if in the or on the circle
            double dy = (y - yPixel), dx = (x - xPixel);
            if (dy*dy + dx*dx > R*R) continue;
            //
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = abs(signedDist);
            // skip the pixels on the line
            if (dist < THICKNESS*SCALE/2) continue;
            //
            thrust::tuple<uchar,uchar,uchar> lab = getColorChannels(F, y, x, width, height);
            int l = thrust::get<0>(lab);
            int a = thrust::get<1>(lab);
            int b = thrust::get<2>(lab);
            minL = min(minL, l);
            minA = min(minA, a);
            minB = min(minB, b);
        }
    }
    for (double y = y1; y <= y2; y += STEP) {
        for (double x = x1; x <= x2; x += STEP) {
            // check if in the or on the circle
            double dy = (y - yPixel), dx = (x - xPixel);
            if (dy*dy + dx*dx > R*R) continue;
            //
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = abs(signedDist);
            // skip the pixels on the line
            if (dist < THICKNESS*SCALE/2) continue;
            //
            double w =  1.0 - sqrt(dx*dx + dy*dy)/R;
            //
            thrust::tuple<uchar,uchar,uchar> lab = getColorChannels(F, y, x, width, height);
            int l = thrust::get<0>(lab);
            int a = thrust::get<1>(lab);
            int b = thrust::get<2>(lab);
            //
            double lContribution = w*(l - minL + OFFSET);
            double abContribution = w*((a - minA + OFFSET) + (b - minB + OFFSET));
            // equalize intensive and non-intensive colors
            if (signedDist > 0) { // the half-circle of the normal vector
                l1 += lContribution;
                ab1 += abContribution;
            } else { // the half-circle opposite to the normal vector
                l2 += lContribution;
                ab2 += abContribution;
            }
        }
    }
    // compute the score (using the logistic function)
    double lRatio = max(l1/l2, l2/l1);
    double abRatio = max(ab1/ab2, ab2/ab1);
    return 1.0 / (1.0 + exp(-(max(lRatio, abRatio) - CAND_RATIO)));
}
