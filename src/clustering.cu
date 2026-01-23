#include "clustering.hpp"

std::vector<bool> solveClustering(const CandidateGraph& G) {
    Graph<> originalGraph(G.n);
    for (const Edge& e : G.edges) {
        originalGraph.insertEdge(e.c1, e.c2);
    }
    //
    CompleteGraph<> liftedGraph(G.n);
    std::vector<double> weights((G.n*(G.n - 1))/2, 0.0);
    for (const Edge& e : G.edges) {
        weights[liftedGraph.findEdge(e.c1, e.c2).second] = e.w;
    }
    // 
    std::vector<char> edgeLabels(liftedGraph.numberOfEdges(), 1);
    multicut_lifted::greedyAdditiveEdgeContraction(originalGraph, liftedGraph, weights, edgeLabels);
    //
    // TODO:
    return std::vector<bool>();
}
