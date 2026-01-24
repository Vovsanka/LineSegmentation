#include "clustering.hpp"

std::vector<char> solveClustering(const CandidateGraph& G) {
    std::size_t n = G.n;
    std::size_t m = G.edges.size();
    //
    Graph<> originalGraph(n);
    for (const Edge& e : G.edges) {
        originalGraph.insertEdge(e.c1, e.c2);
    }
    //
    CompleteGraph<> liftedGraph(n);
    std::vector<double> weights(liftedGraph.numberOfEdges());
    for (const Edge& e : G.edges) {
        weights[liftedGraph.findEdge(e.c1, e.c2).second] = e.w;
    }
    // 
    std::vector<char> allLabels(liftedGraph.numberOfEdges(), 1);
    multicut_lifted::greedyAdditiveEdgeContraction(originalGraph, liftedGraph, weights, allLabels);
    //
    std::vector<char> edgeLabels(m, true);
    for (int k = 0; k < m; k++) {
        const Edge& e = G.edges[k];
        edgeLabels[k] = allLabels[liftedGraph.findEdge(e.c1, e.c2).second];
    }
    return edgeLabels;
}
