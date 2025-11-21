#include "score.hpp"


__host__ __device__ 
thrust::tuple<double,double> directionNormalUnitVector(int d) {
    const double PI = acos(-1.0);
    double rad = d * (PI / DIRECTIONS);
    return thrust::make_tuple(sin(rad), cos(rad));
}

__host__ __device__
double computeLabScore(const uchar* F,
                       double yPixel, double xPixel,
                       int direction, 
                       int width, int height) {
    double area1 = 0, area2 = 0;
    int minL = 255, minA = 255, minB = 255;
    double ABW = (1.0 - LW)/2.0;
    //
    thrust::tuple<double,double> unitNorm = directionNormalUnitVector(direction);
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
            double contribution = LW*w*(l - minL + OFFSET) + ABW*w*((a - minA + OFFSET) + (b - minB + OFFSET));
            // equalize intensive and non-intensive colors
            if (signedDist > 0) { // the half-circle of the normal vector
                area1 += contribution;
            } else { // the half-circle opposite to the normal vector
                area2 += contribution;
            }
        }
    }
    // compute the score (using the logistic function)
    double ratio = max(area1/area2, area2/area1);
    return 1.0 / (1.0 + exp(-(ratio - CAND_RATIO)));
}
