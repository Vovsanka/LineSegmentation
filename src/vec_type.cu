#include "vec_type.hpp"


__host__ __device__
Vec::Vec(double y, double x) {
    this->y = y;
    this->x = x;
}

__host__ __device__   
double Vec::len() {
    return sqrt(y*y + x*x);
}

__host__ __device__    
Vec Vec::subtract(const Vec& otherVec) {
    return Vec(y - otherVec.y, x - otherVec.x);
}

__host__ __device__
Vec Vec::operator*(double k) {
    return Vec(k*y, k*x);
}

__host__ __device__
double Vec::dot(const Vec& otherVec) {
    return x*otherVec.x + y*otherVec.y;
}
