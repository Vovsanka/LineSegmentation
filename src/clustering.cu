#include "clustering.hpp"


std::vector<char> solveClustering(const CandidateGraph& G) {
    std::size_t n = G.n;
    std::size_t m = G.edges.size();
    //
    Graph<> graph(n);
    std::vector<double> weights(m);
    for (int k = 0; k < m; k++) {
        const Edge& e = G.edges[k];
        graph.insertEdge(e.c1, e.c2);
        weights[k] = e.w;
    }
    // 
    std::vector<char> edgeLabels(m);
    multicut::kernighanLin(graph, weights, edgeLabels, edgeLabels);
    // multicut::greedyAdditiveEdgeContraction(graph, weights, edgeLabels);
    //
    return edgeLabels;
}
