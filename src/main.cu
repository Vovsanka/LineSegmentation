#include <iostream> 

#include "lsd.hpp"


int main(int argc, char* argv[]) {
    
    // Check the CUDA devices (GPU support)
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    if (cudaCount > 0) {
        std::cout << "GPU is enabled" << std::endl;
    } else {
        std::cout << "No CUDA devices available" << std::endl;
        return 1;
    }
    
    // set image path and working state directory from program arguments
    if (argc == 3) {
        lsd::imagePath = argv[1];
        lsd::workingStateDir = argv[2];
    } else {
        std::cout << "Error: there must be 2 CLI parameters (image path, working state folder path)" << std::endl;
        return 1;
    }

    // // 1: load the image, convert to LAB, scale down if needed
    // lsd::loadPreprocessImage("original", "preprocessed", "params", false);

    // // 2: compute threshold and iterative candidates (using structure tensor score function or beam score function)
    // lsd::computeThresholdCandidates("preprocessed", "scores", "directions", "t_candidates", false); 
    // lsd::computeIterativeCandidates("preprocessed", "t_candidates", "candidates", false);

    // // 3: build the candidate graph and group the candidates by line segments, extract and reconstruct the line segments
    // lsd::buildCandidateGraph("candidates", "cgraph");
    // lsd::performClustering("cgraph", "labels");
    // lsd::extractLines("candidates", "cgraph", "labels", "lines");
    // lsd::reconstructOriginalLines("params", "lines", "or_lines");

    // extra
    lsd::buildShowStateImages("original", "params", "preprocessed", "scores", "directions", "score-direction", "t_candidates");

    return 0;
}


