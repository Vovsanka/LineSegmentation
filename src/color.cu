#include "color.hpp"


__host__ __device__
thrust::tuple<uchar,uchar,uchar> bicubicInterpolation(const uchar* F,
                                                      float y, float x,
                                                      int width, int height) {
    // clamp‑to‑edge strategy for out of range subpixels
    int x0 = static_cast<int>(floor(x));
    int y0 = static_cast<int>(floor(y));
    float dx = x - x0;
    float dy = y - y0;

    // Inline cubic weight function (Catmull-Rom spline, a = -0.5)
    auto cubicWeight = [] __host__ __device__ (float t) {
        const float a = -0.5;
        t = fabs(t);
        if (t < 1.0f) return (a+2)*t*t*t - (a+3)*t*t + 1;
        else if (t < 2.0) return a*t*t*t - 5*a*t*t + 8*a*t - 4*a;
        else return 0.0f;
    };

    float wx[4], wy[4];
    for (int i=0; i<4; i++) {
        wx[i] = cubicWeight(dx - (i-1));
        wy[i] = cubicWeight(dy - (i-1));
    }

    float C[3] = {0, 0, 0};
    for (int j=0; j<4; j++) {
        for (int i=0; i<4; i++) {
            int xi = min(max(x0+i-1,0), width-1);
            int yj = min(max(y0+j-1,0), height-1);
            int idx = (yj*width + xi)*3;

            uchar c0 = F[idx+0];
            uchar c1 = F[idx+1];
            uchar c2 = F[idx+2];

            float wxy = wx[i]*wy[j];
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
thrust::tuple<uchar,uchar,uchar> getColorChannels(const uchar* F,
                                                  float y, float x,
                                                  int width, int height) {
    int rY = round(y);
    int rX = round(x);
    // image integer pixel case (no need to compute)
    if (0 <= rY && rY < height &&
        0 <= rX && rX < width &&
        fabs(y - rY) <= TOL && fabs(x - rX) <= TOL) {
        int idx = (rY * width + rX) * 3;
        return thrust::make_tuple(F[idx + 0], F[idx + 1], F[idx + 2]);
    }
    // determine the color of the sub-pixel with the bicubic interpolation (possibly out of range)
    return bicubicInterpolation(F, y, x, width, height);
}