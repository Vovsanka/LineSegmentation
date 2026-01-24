#include "lines.hpp"


std::vector<Line> extractLinesFromClusters(
    const std::vector<Cand>& candidates,
    const CandidateGraph& G,
    const std::vector<char>& edgeLabels
) {
    std::vector<std::vector<int>> clusters = retrieveClusters(G, edgeLabels);
    //
    std::vector<Line> lines;
    for (const std::vector<int>& cluster : clusters) {
        if (clusterIsLine(candidates, cluster)) {
            lines.push_back(clusterToLine(candidate, cluster));
        }
    }
    return lines;
}

std::vector<std::vector<int>> retrieveClusters( 
    const CandidateGraph& G,
    const std::vector<char>& edgeLabels
) {
    std::vector<int> clusterMap(G.n, -1);
    int clusterCount = 0;
    for (int k = 0; k < G.edges.size(); k++) {
        if (edgeLabels[k] == 0) {
            const Edge& e = G.edges[k]; 
            if (clusterMap[e.c1] == -1 && clusterMap[e.c2] == -1) {
                clusterMap[e.c1] = clusterMap[e.c2] = clusterCount++; 
            } else if (clusterMap[e.c1] == -1) { // clusterMap[e.c2] != -1
                clusterMap[e.c1] = clusterMap[e.c2];
            } else if (clusterMap[e.c2] == -1) { // clusterMap[e.c1] != -1
                clusterMap[e.c2] = clusterMap[e.c1];
            }
        }
    }
    // gather all the clusters with 2+ nodes
    std::vector<std::vector<int>> clusters(clusterCount, std::vector<int>());
    for (int node = 0; node < G.n; node++) {
        if (clusterMap[node] != -1) {
            clusters[clusterMap[node]].push_back(node);
        }
    }
    return clusters;
}

bool clusterIsLine(
    const std::vector<Cand>& candidates,
    const std::vector<int>& cluster
) {
    // TODO
}

Line clusterToLine() (
    const std::vector<Cand>& candidates,
    const std::vector<int>& cluster
) {
    // TODO: best fit line (opencv maybe?)
}
