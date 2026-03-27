#ifndef CGRAPH_HPP
#define CGRAPH_HPP

#include <algorithm>

#include "config.hpp"
#include "cand_type.hpp"
#include "edge_type.hpp"
#include "cost.hpp"



struct CandidateGraph {
    std::size_t n;
    std::vector<Edge> edges;

    CandidateGraph() = default;

    __host__
    CandidateGraph(std::vector<Cand> candidates);
};


#endif