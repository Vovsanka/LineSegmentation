#include "score.hpp"


__host__ __device__ 
thrust::pair<double,double> directionNormalUnitVector(int d) {
    const double PI = acos(-1.0);
    double rad = d * (PI / DIRECTIONS);
    return thrust::make_pair(sin(rad), cos(rad));
}

__host__ __device__ 
double computeScore(const uchar* F,
                                        double yPixel, double xPixel,
                                        double unitNormY, double unitNormX,
                                        int width, int height) {
    // consider only the pixels s.t. their pixel center (y, x) is on/in the circle with radius R
    int r1, g1, b1, r2, b2, g2;
    r1 = g1 = b1 = r2 = b2 = g2 = 0;
    int minR, minG, minB;
    minR = minG = minB = 255;
    for (int y = max(0, (int)ceilf(yPixel - R)); y <= min(height, (int)floorf(yPixel + R)); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = ceil((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = floor((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = max(0, x1); x <= min(width, x2); x++) {
            int dy = (y - yPixel), dx = (x - xPixel);
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = abs(signedDist);
            // skip the pixels on the line
            if (dist < THICKNESS/2) continue;
            // add pixel to the corresponding half-circle
            auto [r, g, b] = getRgbColors(F, y, x, width, height);
            int w = R*R - (dx*dx + dy*dy);
            minR = min(minR, r);
            minG = min(minG, g);
            minB = min(minB, b);
            if (signedDist >= 0) { // the half-circle of the normal vector
                b1 += w*b;
                g1 += w*g;
                r1 += w*r;
            } else { // the half-circle opposite to the normal vector
                b2 += w*b;
                g2 += w*g;
                r2 += w*r;
            }
        }
    }
    // equalize intensive and non-intensive colors
    for (int y = max(0, (int)ceilf(yPixel - R)); y <= min(height, (int)floorf(yPixel + R)); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = ceilf((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = floorf((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = max(0, x1); x <= min(width, x2); x++) {
            int dy = (y - yPixel), dx = (x - xPixel);
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = abs(signedDist);
            // skip the pixels on the line
            if (dist < 0.1) continue;
            // add pixel to the corresponding half-circle
            int w = R*R - (dx*dx + dy*dy);
            if (signedDist >= 0) {
                b1 += w*(-minB);
                g1 += w*(-minG);
                r1 += w*(-minR);
            } else { // the half-circle opposite to the normal vector
                b2 += w*(-minB);
                g2 += w*(-minG);
                r2 += w*(-minR);
            }
        }
    }
    // compute the score (add offset to reduce the noise sensitivity)
    double area1 = r1 + g1 + b1 + OFFSET;
    double area2 = r2 + g2 + b2 + OFFSET;
    double ratio = max(area1/area2, area2/area1); // avoid div!
    double sqrRatio = ratio*ratio;
    return 1.0 - 1/(sqrRatio*sqrRatio);
}
