#include "cgraph_type.hpp"


__host__ __device__
Edge::Edge(int c1, int c2, double w) {
    this->c1 = c1;
    this->c2 = c2;
    this->w = w;
}

__host__
CandidateGraph::CandidateGraph(const std::vector<Cand>& candidates) {
    this->n = candidates.size();
    this->edges = std::vector<Edge>();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (Cand::dist(candidates[i], candidates[j]) < MAX_PROXIMITY) {
                int cost = computeCandidateCost(candidates, candidates[i], candidates[j]);
                edges.push_back(Edge(i, j, cost));
            }
        }
    }  
    std::cout << n << " " << edges.size() << std::endl;
}
