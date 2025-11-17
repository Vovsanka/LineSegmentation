#include "color.hpp"


__host__ __device__
thrust::tuple<uchar,uchar,uchar> bicubicInterpolation(const uchar* F,
                                                      double y, double x,
                                                      int width, int height) {
    // (clamp‑to‑edge strategy)
    int x0 = static_cast<int>(floor(x));
    int y0 = static_cast<int>(floor(y));
    double dx = x - x0;
    double dy = y - y0;

    // Inline cubic weight function (Catmull-Rom spline, a = -0.5)
    auto cubicWeight = [] __host__ __device__ (double t) {
        const double a = -0.5;
        t = fabs(t);
        if (t < 1.0) return (a+2)*t*t*t - (a+3)*t*t + 1;
        else if (t < 2.0) return a*t*t*t - 5*a*t*t + 8*a*t - 4*a;
        else return 0.0;
    };

    double wx[4], wy[4];
    for (int i=0; i<4; i++) {
        wx[i] = cubicWeight(dx - (i-1));
        wy[i] = cubicWeight(dy - (i-1));
    }

    double R=0, G=0, B=0;
    for (int j=0; j<4; j++) {
        for (int i=0; i<4; i++) {
            int xi = min(max(x0+i-1,0), width-1);
            int yj = min(max(y0+j-1,0), height-1);
            int idx = (yj*width + xi)*3;

            uchar r = F[idx+0];
            uchar g = F[idx+1];
            uchar b = F[idx+2];

            double wxy = wx[i]*wy[j];
            R += r * wxy;
            G += g * wxy;
            B += b * wxy;
        }
    }

    uchar rOut = static_cast<uchar>(max(0.0, min(255.0, round(R))));
    uchar gOut = static_cast<uchar>(max(0.0, min(255.0, round(G))));
    uchar bOut = static_cast<uchar>(max(0.0, min(255.0, round(B))));

    return thrust::make_tuple(rOut, gOut, bOut);
}

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getRgbColors(const uchar* F,
                                              double y, double x,
                                              int width, int height) {
    int rY = round(y);
    int rX = round(x);
    // integer pixel case (no need to compute)
    if (fabs(y - rY) <= TOL && fabs(x - rX) <= TOL) {
        int idx = (rY * width + rX) * 3;
        return thrust::make_tuple(F[idx + 2], F[idx + 1], F[idx]);
    }
    // determine the color of the sub-pixel with the bicubic interpolation
    return bicubicInterpolation(F, y, x, width, height);
}