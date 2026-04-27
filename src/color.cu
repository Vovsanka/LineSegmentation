#include "color.hpp"

__host__ __device__
uchar bicubicInterpolation1(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int width, int height
) {
    // Base integer pixel
    int x0 = static_cast<int>(floor(x));
    int y0 = static_cast<int>(floor(y));
    double dx = x - x0;
    double dy = y - y0;
    // Catmull-Rom spline, a = -0.5
    auto cubicWeight = [] __host__ __device__ (double t) {
        const double a = -0.5;
        t = fabs(t);
        if (t < 1.0)      return (a+2)*t*t*t - (a+3)*t*t + 1.0;
        else if (t < 2.0) return a*t*t*t - 5*a*t*t + 8*a*t - 4*a;
        else              return 0.0;
    };
    //
    double wx[4], wy[4];
    for (int i = 0; i < 4; ++i) {
        wx[i] = cubicWeight(dx - (i - 1));
        wy[i] = cubicWeight(dy - (i - 1));
    }
    //
    double C0 = 0.0;
    for (int j = 0; j < 4; ++j) {
        int yj = min(max(y0 + j - 1, 0), height - 1);
        const uchar* row = F + yj * Fstep;
        for (int i = 0; i < 4; ++i) {
            int xi = min(max(x0 + i - 1, 0), width - 1);
            uchar c0 = row[xi];
            //
            double wxy = wx[i] * wy[j];
            C0 += c0 * wxy;
        }
    }
    //
    auto clamp255 = [] __host__ __device__ (double v) {
        v = round(v);
        if (v < 0.0)   v = 0.0;
        if (v > 255.0) v = 255.0;
        return static_cast<uchar>(v);
    };
    //
    uchar c0norm = clamp255(C0);
    return c0norm;
}

__host__ __device__
thrust::tuple<uchar,uchar,uchar> bicubicInterpolation3(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int width, int height
) {
    // Base integer pixel
    int x0 = static_cast<int>(floor(x));
    int y0 = static_cast<int>(floor(y));
    double dx = x - x0;
    double dy = y - y0;
    // Catmull-Rom spline, a = -0.5
    auto cubicWeight = [] __host__ __device__ (double t) {
        const double a = -0.5;
        t = fabs(t);
        if (t < 1.0)      return (a+2)*t*t*t - (a+3)*t*t + 1.0;
        else if (t < 2.0) return a*t*t*t - 5*a*t*t + 8*a*t - 4*a;
        else              return 0.0;
    };
    //
    double wx[4], wy[4];
    for (int i = 0; i < 4; ++i) {
        wx[i] = cubicWeight(dx - (i - 1));
        wy[i] = cubicWeight(dy - (i - 1));
    }
    //
    double C[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < 4; ++j) {
        int yj = min(max(y0 + j - 1, 0), height - 1);
        const uchar* row = F + yj * Fstep;
        for (int i = 0; i < 4; ++i) {
            int xi = min(max(x0 + i - 1, 0), width - 1);
            int base = xi * 3;
            uchar c0 = row[base + 0];
            uchar c1 = row[base + 1];
            uchar c2 = row[base + 2];
            //
            double wxy = wx[i] * wy[j];
            C[0] += c0 * wxy;
            C[1] += c1 * wxy;
            C[2] += c2 * wxy;
        }
    }
    //
    auto clamp255 = [] __host__ __device__ (double v) {
        v = round(v);
        if (v < 0.0)   v = 0.0;
        if (v > 255.0) v = 255.0;
        return static_cast<uchar>(v);
    };
    //
    uchar c0norm = clamp255(C[0]);
    uchar c1norm = clamp255(C[1]);
    uchar c2norm = clamp255(C[2]);
    return thrust::make_tuple(c0norm, c1norm, c2norm);
}

__host__ __device__
uchar getGrayColor(
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
        return row[rX];
    }
    // determine the color of the sub-pixel with the bicubic interpolation (possibly out of range)
    return bicubicInterpolation1(F, Fstep, y, x, width, height);
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
        int base = rX * 3; 
        uchar c0 = row[base + 0]; 
        uchar c1 = row[base + 1]; 
        uchar c2 = row[base + 2]; 
        return thrust::make_tuple(c0, c1, c2);
    }
    // determine the color of the sub-pixel with the bicubic interpolation (possibly out of range)
    return bicubicInterpolation3(F, Fstep, y, x, width, height);
}


thrust::tuple<uchar, uchar, uchar> hsvToRgb(double h, double s, double v) {
    double r, g, b;

    int i = int(h * 6.0);
    double f = h * 6.0 - i;
    double p = v * (1.0 - s);
    double q = v * (1.0 - f * s);
    double t = v * (1.0 - (1.0 - f) * s);

    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }

    // convert to uchar safely
    auto toU8 = [](double x) {
        x = std::max(0.0, std::min(1.0, x));  // clamp
        return static_cast<uchar>(x * 255.0 + 0.5); // round
    };

    return thrust::make_tuple(toU8(r), toU8(g), toU8(b));
}

__host__
thrust::tuple<uchar,uchar,uchar> getDirColorRgb(int dir) {
    double rad = getRad(2*dir);
    double hue = fmod(rad / (2*PI), 1.0);
    if (hue < 0) hue += 1.0;
    return hsvToRgb(hue, 1.0, 1.0);
}

__host__
thrust::tuple<uchar,uchar,uchar> getDirColorLab(int dir) {
    thrust::tuple<uchar,uchar,uchar> rgb = getDirColorRgb(dir);
    // convert LAB to RGB
    cv::Mat RGB(1, 1, CV_8UC3, cv::Vec3b(
        thrust::get<0>(rgb),
        thrust::get<1>(rgb),
        thrust::get<2>(rgb)
    ));
    cv::Mat LAB;
    cv::cvtColor(RGB, LAB, cv::COLOR_RGB2Lab);
    cv::Vec3b pix = LAB.at<cv::Vec3b>(0, 0);
    return thrust::make_tuple(pix[0], pix[1], pix[2]);
}

cv::Mat convertBGRtoLab(const cv::Mat& cpuF) {
    cv::Mat labF;
    cv::cvtColor(cpuF, labF, cv::COLOR_BGR2Lab);
    return labF;
}

cv::Mat convertBGRtoGrayscale(const cv::Mat& cpuF) {
    cv::Mat grayF;
    cv::cvtColor(cpuF, grayF, cv::COLOR_BGR2GRAY);
    return grayF;
}
