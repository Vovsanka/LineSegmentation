#include "cgraph_type.hpp"


__host__
CandidateGraph::CandidateGraph(std::vector<Cand> candidates) {
    this->n = candidates.size();
    this->edges = std::vector<Edge>();
    // construct cost edges from candidates
    // sort the candidates with their index by their position (y, x)
    std::vector<std::pair<Cand,int>> candInd(n);
    for (int k = 0; k < n; k++) {
        candInd[k] = std::make_pair(candidates[k], k);
    }
    std::sort(
        std::begin(candInd), std::end(candInd),
        [](const std::pair<Cand,int>& c1, const std::pair<Cand,int>& c2) {
            if (c1.first.y == c2.first.y) return (c1.first.x < c2.first.x);
            return (c1.first.y < c2.first.y);
        }
    );
    // save sorted candidates 
    for (int k = 0; k < n; k++) {
        candidates[k] = candInd[k].first;
    }
    // compute the candidate costs wrt the candidate indices
    for (int i = 0; i < n; i++) {
        Cand& cand1 = candInd[i].first;
        int ind1 = candInd[i].second;
        for (int j = i + 1; j < n; j++) {
            Cand& cand2 = candInd[j].first;
            int ind2 = candInd[j].second;
            if (Cand::dist(cand1, cand2) <= CONNECTION_RADIUS) {
                double cost = computeCandidateCost(candidates, cand1, cand2);
                if (abs(cost) > TOL) {
                    edges.push_back(Edge(ind1, ind2, cost));
                }
            }
        }
    }  
}
