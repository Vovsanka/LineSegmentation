#include "cgraph_type.hpp"



CandidateGraph::CandidateGraph(const std::vector<Cand>& candidates) {
    this->n = candidates.size();
    this->edges = std::vector<Edge>();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (Cand::dist(candidates[i], candidates[j]) <= CONNECTION_RADIUS) {
                double cost = computeCandidateCost(candidates, candidates[i], candidates[j]);
                if (abs(cost) > TOL) {
                    edges.push_back(Edge(i, j, cost));
                }
            }
        }
    }  
}
