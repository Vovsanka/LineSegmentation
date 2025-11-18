#include "score.hpp"


__host__ __device__ 
thrust::tuple<double,double> directionNormalUnitVector(int d) {
    const double PI = acos(-1.0);
    double rad = d * (PI / DIRECTIONS);
    return thrust::make_tuple(sin(rad), cos(rad));
}

__host__ __device__
double computeScore(const uchar* F,
                    double yPixel, double xPixel,
                    int direction, 
                    int width, int height) {
    int r1, g1, b1, c1, r2, b2, g2, c2;
    r1 = g1 = b1 = c1 = r2 = b2 = g2 = c2 = 0;
    int minR, minG, minB;
    minR = minG = minB = 255;
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
            if (dist < THICKNESS/2) continue;
            //
            thrust::tuple<uchar,uchar,uchar> rgb = getRgbColors(F, y, x, width, height);
            int r = thrust::get<0>(rgb);
            int g = thrust::get<1>(rgb);
            int b = thrust::get<2>(rgb);
            minR = min(minR, r);
            minG = min(minG, g);
            minB = min(minB, b);
            //
            if (signedDist > 0) { // the half-circle of the normal vector
                c1++;
                b1 += b;
                g1 += g;
                r1 += r;
            } else { // the half-circle opposite to the normal vector
                c2++;
                b2 += b;
                g2 += g;
                r2 += r;
            }
        }
    }
    // equalize intensive and non-intensive colors
    double offset = -(minR + minG + minB) + OFFSET;
    double area1 = 1.0*(r1 + g1 + b1)/c1 + offset;
    double area2 = 1.0*(r2 + g2 + b2)/c2 + offset;
    double ratio = max(area1/area2, area2/area1);
    // compute the score (using the logistic function)
    return 1.0 / (1.0 + exp(-(ratio - CAND_RATIO)));
}
