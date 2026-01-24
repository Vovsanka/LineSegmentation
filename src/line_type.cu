#include "line_type.hpp"

__host__ __device__
Line::Line(double y1, double x1, double y2, double x2) {
    this->y1 = y1;
    this->x1 = x1;
    this->y2 = y2;
    this->x2 = x2;
}