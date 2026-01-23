#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#include <opencv2/opencv.hpp>

#include "andres-graph/graph.hxx"
#include "andres-graph/complete-graph.hxx"
#include "andres-graph/greedy-additive.hxx"

#include "cgraph.hpp"

using namespace andres::graph;


std::vector<bool> solveClustering(const CandidateGraph& G);


#endif