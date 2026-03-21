#include "edge_type.hpp"


__host__ __device__
Edge::Edge(int c1, int c2, double w) {
    this->c1 = c1;
    this->c2 = c2;
    this->w = w;
}