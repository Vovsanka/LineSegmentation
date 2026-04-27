#include "clustering.hpp"


std::vector<std::vector<int>> solveClustering(const CandidateGraph& G, std::string method) {
    using namespace andres::graph;
    using namespace andres::graph::multicut;

    const int nOriginal = G.n;
    const int mOriginal = G.edges.size();

    Graph<> graph(nOriginal);
    std::vector<double> weights(mOriginal);

    for (int k = 0; k < mOriginal; ++k) {
        const Edge& e = G.edges[k];
        graph.insertEdge(e.c1, e.c2);
        weights[k] = e.w;
    }

    auto reducedInstance = preprocessing(graph, weights);

    Graph<> redGraph = std::get<0>(reducedInstance);
    std::vector<double> redWeights = std::get<1>(reducedInstance);
    auto& mapping = std::get<4>(reducedInstance); 

    const int nRed = redGraph.numberOfVertices();
    const int mRed = redGraph.numberOfEdges();

    std::vector<char> edgeLabels(mRed);

    if (method == "GA+KL") {
        greedyAdditiveEdgeContraction(redGraph, redWeights, edgeLabels);
        kernighanLin(redGraph, redWeights, edgeLabels, edgeLabels);

    } else if (method == "MWS+KL") {
        mutexWatershed(redGraph, redWeights, edgeLabels);
        kernighanLin(redGraph, redWeights, edgeLabels, edgeLabels);

    } else if (method == "MWS") {
        mutexWatershed(redGraph, redWeights, edgeLabels);

    } else if (method == "GA") {
        greedyAdditiveEdgeContraction(redGraph, redWeights, edgeLabels);

    } else if (method == "KL") {
        kernighanLin(redGraph, redWeights, edgeLabels, edgeLabels);
    } else {
        throw std::runtime_error("Unknown clustering method!");
    }

    // disjoint set union
    std::vector<int> parent(nRed);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&](int x) {
        while (parent[x] != x) x = parent[x];
        return x;
    };

    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[b] = a;
    };

    for (int e = 0; e < mRed; ++e) {
        if (edgeLabels[e] == 0) { 
            int u = redGraph.vertexOfEdge(e, 0);
            int v = redGraph.vertexOfEdge(e, 1);
            unite(u, v);
        }
    }

    std::vector<int> labelOriginal(nOriginal, -1);

    for (int rv = 0; rv < mapping.size(); ++rv) {
        int comp = find(static_cast<int>(rv));
        int ov = mapping[rv]; // original vertex
        labelOriginal[ov] = comp;
    }

    std::unordered_map<int, std::vector<int>> buckets;

    for (int i = 0; i < nOriginal; ++i) {
        int c = labelOriginal[i];
        if (c >= 0)
            buckets[c].push_back(i);
    }

    std::vector<std::vector<int>> clusters;
    clusters.reserve(buckets.size());
    for (auto& kv : buckets) {
        clusters.push_back(kv.second);
    }

    return clusters;
}
