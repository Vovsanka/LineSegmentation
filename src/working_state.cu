#include "working_state.hpp"


void saveMatrix(const cv::Mat& M, std::string name) {
    std::ofstream out(name + ".bin", std::ios::binary);
    int rows = M.rows, cols = M.cols, type = M.type();
    out.write((char*)&rows, sizeof(int)); 
    out.write((char*)&cols, sizeof(int)); 
    out.write((char*)&type, sizeof(int)); 
    out.write((char*)M.data, M.total()*M.elemSize());
}

cv::Mat loadMatrix(std::string name) {
    std::ifstream in(name + ".bin", std::ios::binary); 
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
    std::ofstream out(name + ".bin", std::ios::binary);
    std::size_t count = candidates.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (count > 0) {
        out.write(reinterpret_cast<const char*>(candidates.data()), count*sizeof(Cand)); 
    }
}

std::vector<Cand> loadCandidates(std::string name) {
    std::ifstream in(name + ".bin", std::ios::binary);
    std::size_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<Cand> candidates(count);
    if (count > 0) { 
        in.read(reinterpret_cast<char*>(candidates.data()), count * sizeof(Cand)); 
    } 
    return candidates;
}
