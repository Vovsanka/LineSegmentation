#include "directions.hpp"

__host__ __device__
double getRad(int d) { // returns [0, 2*PI)
    d %= 2*DIRECTIONS;
    return d*PI/DIRECTIONS;
}

__host__ __device__ 
Vec getUnitVector(int d) { // (y, x) // d in [0, 2*DIRECTIONS)
    float rad = static_cast<float>(getRad(d));
    return Vec(sinf(rad), cosf(rad));
}

__host__ __device__
int getOrthogonalDirection(int d) { // d in [0, 2*DIRECTIONS)
    return (d + DIRECTIONS/2) % (2*DIRECTIONS);
}

__host__ __device__
int getOppositeDirection(int d) { // d in [0, 2*DIRECTIONS)
    return (d + DIRECTIONS) % (2*DIRECTIONS);
}

__host__ __device__ 
Vec getOrthogonalUnitVector(int d) { // (y, x) // d in [0, 2*DIRECTIONS)
    return getUnitVector(getOrthogonalDirection(d));
}

__host__ __device__
int getDirDifference(int d1, int d2) { // d1, d2 in [0, DIRECTIONS)
    int dirSmaller = min(d1, d2);
    int dirLarger = max(d1, d2);
    return min(dirLarger - dirSmaller, DIRECTIONS - dirLarger + dirSmaller);
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

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getDirColorRgb(int dir) {
    double rad = getRad(2*dir);
    double hue = fmod(rad / (2*PI), 1.0);
    if (hue < 0) hue += 1.0;
    return hsvToRgb(hue, 1.0, 1.0);
}

__host__ __device__
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

