#include "types.hpp"


__host__ __device__
Vec::Vec(double y, double x) {
    this->y = y;
    this->x = x;
}

__host__ __device__
Cand::Cand(double y, double x, int dir) {
    this->y = y;
    this->x = x;
    this->dir = dir;
}