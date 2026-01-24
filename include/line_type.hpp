#ifndef LINE_TYPE_HPP
#define LINE_TYPE_HPP

#include <opencv2/opencv.hpp>


struct Line {
    double y1, x1, y2, x2;

    Line() = default;

    __host__ __device__
    Line(double y1, double x1, double y2, double x2);
};


#endif