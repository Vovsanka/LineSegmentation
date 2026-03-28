#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#include <numeric>
#include <unordered_map>

#include <andres-graph/graph.hxx>
#include <andres-graph/preprocessing.hxx>
#include <andres-graph/greedy-additive.hxx>
#include <andres-graph/mutex-watershed.hxx>
#include <andres-graph/kernighan-lin.hxx>

#include "cgraph_type.hpp"

using namespace andres::graph;

std::vector<std::vector<int>> solveClustering(const CandidateGraph& G);


#endif