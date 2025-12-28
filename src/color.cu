#include "color.hpp"


__host__ __device__
thrust::tuple<uchar,uchar,uchar> bicubicInterpolation(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int width, int height
) {
    // clamp‑to‑edge strategy for out of range subpixels
    int x0 = static_cast<int>(floor(x));
    int y0 = static_cast<int>(floor(y));
    double dx = x - x0;
    double dy = y - y0;

    // atmull-Rom spline, a = -0.5
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

    double C[3] = {0, 0, 0};
    for (int j=0; j<4; j++) {
        for (int i=0; i<4; i++) {
            int xi = min(max(x0+i-1,0), width-1);
            int yj = min(max(y0+j-1,0), height-1);
            //
            const uchar* row = F + yj * Fstep;
            const uchar3* row3 = reinterpret_cast<const uchar3*>(row);
            uchar3 pix = row3[xi];
            uchar c0 = pix.x;
            uchar c1 = pix.y;
            uchar c2 = pix.z;
            //
            double wxy = wx[i]*wy[j];
            C[0] += c0 * wxy;
            C[1] += c1 * wxy;
            C[2] += c2 * wxy;
        }
    }

    uchar c0norm = static_cast<uchar>(max(0.0, min(255.0, round(C[0]))));
    uchar c1norm = static_cast<uchar>(max(0.0, min(255.0, round(C[1]))));
    uchar c2norm = static_cast<uchar>(max(0.0, min(255.0, round(C[2]))));

    return thrust::make_tuple(c0norm, c1norm, c2norm);
}

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getColorChannels(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int width, int height
) {
    int rY = round(y);
    int rX = round(x);
    // image integer pixel case (no need to compute)
    if (0 <= rY && rY < height &&
        0 <= rX && rX < width &&
        fabs(y - rY) <= TOL && fabs(x - rX) <= TOL) {
        const uchar* row = F + rY * Fstep;
        const uchar3* row3 = reinterpret_cast<const uchar3*>(row);
        uchar3 pix = row3[rX];
        return thrust::make_tuple(pix.x, pix.y, pix.z);
    }
    // determine the color of the sub-pixel with the bicubic interpolation (possibly out of range)
    return bicubicInterpolation(F, Fstep, y, x, width, height);
}