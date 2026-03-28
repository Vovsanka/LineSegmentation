#include "cgraph_type.hpp"


__host__
CandidateGraph::CandidateGraph(std::vector<Cand> candidates) {
    this->n = candidates.size();
    this->edges = std::vector<Edge>();
    // construct cost edges from candidates
    constructCostEdges(candidates);
    // reduce the candidate graph
    reduce();
}

__host__
void CandidateGraph::constructCostEdges(std::vector<Cand>& candidates) {
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

void CandidateGraph::reduce() {
     // transform to the library graph
    Graph<> graph(n);
    std::vector<double> weights(edges.size());
    for (int k = 0; k < edges.size(); k++) {
        const Edge& e = edges[k];
        graph.insertEdge(e.c1, e.c2);
        weights[k] = e.w;
    }
    // graph reduction
    auto reducedInstance = andres::graph::multicut::preprocessing(graph, weights);
    graph = std::get<0>(reducedInstance);
    weights = std::get<1>(reducedInstance);
    // reconstruct the reduced candidate graph
    edges = std::vector<Edge>(graph.numberOfEdges());
    for (size_t k = 0; k < graph.numberOfEdges(); k++) {
        edges[k] = Edge(graph.vertexOfEdge(k, 0), graph.vertexOfEdge(k, 1), weights[k]);
    }
}
