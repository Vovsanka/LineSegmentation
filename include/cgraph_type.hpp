#ifndef CGRAPH_HPP
#define CGRAPH_HPP

#include <algorithm>

#include <andres-graph/graph.hxx>
#include <andres-graph/preprocessing.hxx>

#include "config.hpp"
#include "cand_type.hpp"
#include "edge_type.hpp"
#include "cost.hpp"


using namespace andres::graph;


struct CandidateGraph {
    std::size_t n;
    std::vector<Edge> edges;

    CandidateGraph() = default;

    __host__
    CandidateGraph(std::vector<Cand> candidates);

    __host__
    void constructCostEdges(std::vector<Cand>& candidates);

    __host__
    void reduce(); 
};


#endif