#ifndef CGRAPH_HPP
#define CGRAPH_HPP

#include "config.hpp"
#include "cand_type.hpp"
#include "cost.hpp"


struct Edge {
    int c1, c2;
    double w;

    Edge() = default;

    __host__ __device__
    Edge(int c1, int c2, double w);
};

struct CandidateGraph {
    int n;
    std::vector<Edge> edges;

    CandidateGraph() = default;

    __host__
    CandidateGraph(const std::vector<Cand>& candidates);
};


#endif