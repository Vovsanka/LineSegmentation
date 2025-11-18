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
    double r1, g1, b1, r2, b2, g2;
    r1 = g1 = b1 = r2 = b2 = g2 = 0;
    double minR, minG, minB;
    minR = minG = minB = 255;
    //
    thrust::tuple<double,double> unitNorm = directionNormalUnitVector(direction);
    double unitNormY = thrust::get<0>(unitNorm);
    double unitNormX = thrust::get<1>(unitNorm); 
    //
    double y1 = fmax(0.0, yPixel - R);
    double y2 = fmin(height - 1.0, yPixel + R);
    for (double y = y1; y <= y2; y += STEP) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= R^2
        double x1 = fmax(0.0, xPixel - R);
        double x2 = fmin(width - 1.0, xPixel + R);
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
            minR = fmin(minR, r);
            minG = fmin(minG, g);
            minB = fmin(minB, b);
            //
            if (signedDist > 0) { // the half-circle of the normal vector
                b1 += b;
                g1 += g;
                r1 += r;
            } else { // the half-circle opposite to the normal vector
                b2 += b;
                g2 += g;
                r2 += r;
            }
        }
    }
    // equalize intensive and non-intensive colors
    for (double y = y1; y <= y2; y += STEP) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double x1 = fmax(0.0, xPixel - R);
        double x2 = fmin(width - 1.0, xPixel + R);
        for (double x = x1; x <= x2; x += STEP) {
            // check if in the or on the circle
            double dy = (y - yPixel), dx = (x - xPixel);
            if (dy*dy + dx*dx > R*R) continue;
            //
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = abs(signedDist);
            // skip the pixels on the line
            if (dist < THICKNESS/2) continue;
            // add pixel to the corresponding half-circle
            if (signedDist > 0) {
                b1 -= minB;
                g1 -= minG;
                r1 -= minR;
            } else { // the half-circle opposite to the normal vector
                b2 -= minB;
                g2 -= minG;
                r2 -= minR;
            }
        }
    }
    // compute the score (add offset to reduce the noise sensitivity)
    double area1 = r1 + g1 + b1 + OFFSET;
    double area2 = r2 + g2 + b2 + OFFSET;
    double ratio = max(area1/area2, area2/area1); // avoid div!
    // logistic function
    return 1.0 / (1.0 + exp(-(ratio - CAND_RATIO)));
}
