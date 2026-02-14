#include "working_state.hpp"


void saveImageParams(
    int originalWidth, int originalHeight,
    int width, int height,
    std::string name
) {
    std::ofstream out(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(&originalWidth), sizeof(originalWidth));
    out.write(reinterpret_cast<const char*>(&originalHeight), sizeof(originalHeight));
    out.write(reinterpret_cast<const char*>(&width), sizeof(width));
    out.write(reinterpret_cast<const char*>(&height), sizeof(height));
}

std::tuple<int,int,int,int> loadImageParams(std::string name) {
    std::ifstream in(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    int originalWidth = 0, originalHeight = 0;
    int width = 0, height = 0;
    in.read(reinterpret_cast<char*>(&originalWidth), sizeof(originalWidth));
    in.read(reinterpret_cast<char*>(&originalHeight), sizeof(originalHeight));
    in.read(reinterpret_cast<char*>(&width), sizeof(width));
    in.read(reinterpret_cast<char*>(&height), sizeof(height));
    return std::make_tuple(originalWidth, originalHeight, width, height);
}

void saveMatrix(const cv::Mat& M, std::string name) {
    std::ofstream out(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    int rows = M.rows, cols = M.cols, type = M.type();
    out.write((char*)&rows, sizeof(int)); 
    out.write((char*)&cols, sizeof(int)); 
    out.write((char*)&type, sizeof(int)); 
    out.write((char*)M.data, M.total()*M.elemSize());
}

cv::Mat loadMatrix(std::string name) {
    std::ifstream in(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary); 
    int rows, cols, type; 
    in.read((char*)&rows, sizeof(int)); 
    in.read((char*)&cols, sizeof(int)); 
    in.read((char*)&type, sizeof(int)); 
    //
    cv::Mat M(rows, cols, type); 
    in.read((char*)M.data, M.total()*M.elemSize());
    return M;
}

void saveCandidates(const std::vector<Cand>& candidates, std::string name) {
    std::ofstream out(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    std::size_t count = candidates.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (count > 0) {
        out.write(reinterpret_cast<const char*>(candidates.data()), count*sizeof(Cand)); 
    }
}

std::vector<Cand> loadCandidates(std::string name) {
    std::ifstream in(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    std::size_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<Cand> candidates(count);
    if (count > 0) { 
        in.read(reinterpret_cast<char*>(candidates.data()), count*sizeof(Cand)); 
    } 
    return candidates;
}

void saveCandidateGraph(const CandidateGraph& G, std::string name) {
    std::ofstream out(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    std::size_t vertexCount = G.n;
    std::size_t edgeCount = G.edges.size();
    out.write(reinterpret_cast<const char*>(&vertexCount), sizeof(vertexCount));
    out.write(reinterpret_cast<const char*>(&edgeCount), sizeof(edgeCount));
    if (edgeCount > 0) {
        out.write(reinterpret_cast<const char*>(G.edges.data()), edgeCount*sizeof(Edge)); 
    }
}

CandidateGraph loadCandidateGraph(std::string name) {
    std::ifstream in(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    std::size_t vertexCount = 0;
    std::size_t edgeCount = 0;
    in.read(reinterpret_cast<char*>(&vertexCount), sizeof(vertexCount));
    in.read(reinterpret_cast<char*>(&edgeCount), sizeof(edgeCount));
    std::vector<Edge> edges(edgeCount);
    if (edgeCount > 0) { 
        in.read(reinterpret_cast<char*>(edges.data()), edgeCount*sizeof(Edge)); 
    } 
    //
    CandidateGraph G;
    G.n = vertexCount;
    G.edges = edges;
    return G;
}

void saveEdgeLabels(const std::vector<char>& edgeLabels, std::string name) {
    std::ofstream out(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    std::size_t edgeCount = edgeLabels.size();
    out.write(reinterpret_cast<const char*>(&edgeCount), sizeof(edgeCount));
    if (edgeCount > 0) {
        out.write(reinterpret_cast<const char*>(edgeLabels.data()), edgeCount); 
    }
}

std::vector<char> loadEdgeLabels(std::string name) {
    std::ifstream in(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    std::size_t edgeCount = 0;
    in.read(reinterpret_cast<char*>(&edgeCount), sizeof(edgeCount));
    std::vector<char> edgeLabels(edgeCount);
    if (edgeCount > 0) { 
        in.read(reinterpret_cast<char*>(edgeLabels.data()), edgeCount); 
    } 
    return edgeLabels;
}

void saveLines(const std::vector<Line>& lines, std::string name) {
    std::ofstream out(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    std::size_t lineCount = lines.size();
    out.write(reinterpret_cast<const char*>(&lineCount), sizeof(lineCount));
    if (lineCount > 0) {
        out.write(reinterpret_cast<const char*>(lines.data()), lineCount*sizeof(Line)); 
    }
}

std::vector<Line> loadLines(std::string name) {
    std::ifstream in(WORKING_STATE_DIR/(name + ".bin"), std::ios::binary);
    std::size_t lineCount = 0;
    in.read(reinterpret_cast<char*>(&lineCount), sizeof(lineCount));
    std::vector<Line> lines(lineCount);
    if (lineCount > 0) { 
        in.read(reinterpret_cast<char*>(lines.data()), lineCount*sizeof(Line)); 
    } 
    return lines;
}
