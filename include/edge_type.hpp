#ifndef EDGE_TYPE_HPP
#define EDGE_TYPE_HPP


struct Edge {
    int c1, c2;
    double w;

    Edge() = default;

    __host__ __device__
    Edge(int c1, int c2, double w);
};

#endif