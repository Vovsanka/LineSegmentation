#include "clustering.hpp"


// Assumed:
// struct Edge { int c1, c2; double w; };
// struct CandidateGraph { int n; std::vector<Edge> edges; };
// using Graph = andres::graph::Graph<>;

std::vector<std::vector<int>> solveClustering(const CandidateGraph& G) {
    using namespace andres::graph;
    using namespace andres::graph::multicut;

    // -------------------------------
    // 1. Build initial graph
    // -------------------------------
    const int nOriginal = G.n;
    const int mOriginal = G.edges.size();

    Graph<> graph(nOriginal);
    std::vector<double> weights(mOriginal);

    for (int k = 0; k < mOriginal; ++k) {
        const Edge& e = G.edges[k];
        graph.insertEdge(e.c1, e.c2);
        weights[k] = e.w;
    }

    // -------------------------------
    // 2. Preprocessing
    //
    // tuple layout of preprocessing(graph, weights):
    //  0: reduced graph (Graph<>)
    //  1: reduced weights (std::vector<double>)
    //  2: lower bound (double)
    //  3: fixed edge labels (std::vector<std::pair<std::pair<size_t,size_t>,char>>)
    //  4: mapping (std::vector<size_t>)  // reduced vertex -> original vertex
    // -------------------------------
    auto reducedInstance = preprocessing(graph, weights);

    Graph<> redGraph               = std::get<0>(reducedInstance);
    std::vector<double> redWeights = std::get<1>(reducedInstance);
    auto& mapping                  = std::get<4>(reducedInstance); // vector<size_t>

    const int nRed = redGraph.numberOfVertices();
    const int mRed = redGraph.numberOfEdges();

    // -------------------------------
    // 3. Greedy-additive + KL on reduced graph
    // -------------------------------
    std::vector<char> edgeLabels(mRed);

    // greedyAdditiveEdgeContraction(redGraph, redWeights, edgeLabels);
    mutexWatershed(redGraph, redWeights, edgeLabels);
    kernighanLin(redGraph, redWeights, edgeLabels, edgeLabels);

    // -------------------------------
    // 4. Edge labels -> components (union-find on reduced graph)
    // -------------------------------
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
        if (edgeLabels[e] == 0) { // 0 = uncut -> same component
            int u = redGraph.vertexOfEdge(e, 0);
            int v = redGraph.vertexOfEdge(e, 1);
            unite(u, v);
        }
    }

    // -------------------------------
    // 5. Lift reduced components -> original nodes
    // mapping[rv] = original vertex index
    // -------------------------------
    std::vector<int> labelOriginal(nOriginal, -1);

    for (int rv = 0; rv < mapping.size(); ++rv) {
        int comp = find(static_cast<int>(rv));
        int ov = mapping[rv]; // original vertex
        labelOriginal[ov] = comp;
    }

    // -------------------------------
    // 6. Build clusters: vector<vector<size_t>>
    // -------------------------------
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
