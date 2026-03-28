#include <iostream> 

#include "lsd.hpp"


int main(int argc, char* argv[]) {
    
    /// Check the CUDA devices (GPU support)
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    if (cudaCount > 0) {
        std::cout << "GPU is enabled" << std::endl;
    } else {
        std::cout << "No CUDA devices available" << std::endl;
        return 1;
    }
    
    /// Set image path and working state directory from program arguments
    if (argc >= 3) {
        lsd::imagePath = argv[1];
        lsd::workingStateDir = argv[2];
    } else {
        std::cout << "Error: there must be 2 CLI parameters (image path, working state folder path)" << std::endl;
        return 1;
    }

    // structural settings
    bool runLoadPreprocessImage = true; 
    bool runComputeThresholdCandidates = true;
    bool runComputeIterativeCandidates = true;
    bool runBuildCandidateGraph = true;
    bool runPerformClustering = true;
    bool runExtractLines = true;
    bool show = false;
    // content settings
    bool beams = true; // beams score function or structure tensor score functin
    bool iterative = true; // iterative candidates or threshold candidates
    std::string clusteringMethod = ""; // "GA+KL" (default), "MWS+KL", "MWS", "GA", "KL" 

    // retrieve setting from CLI
    for (int k = 3; k < argc; k++) {
        // set beams
        if (argv[k] == "--bm") beams = true;
        if (argv[k] == "--st") beams = false;
        // set iterative
        if (argv[k] == "--it") iterative = true;
        if (argv[k] == "--th") iterative = false;
        // set clustering method
        if (argv[k] == "--ga-kl") clusteringMethod = "GA+KL";
        if (argv[k] == "--mws-kl") clusteringMethod = "MWS+KL";
        if (argv[k] == "--mws") clusteringMethod = "MWS";
        if (argv[k] == "--ga") clusteringMethod = "GA";
        if (argv[k] == "--kl") clusteringMethod = "KL";
        // 
        if (argv[k] == "--on-lp") runLoadPreprocessImage = true;
        if (argv[k] == "--off-lp") runLoadPreprocessImage = false;
        //
        if (argv[k] == "--on-tc") runComputeThresholdCandidates = true;
        if (argv[k] == "--off-tc") runComputeThresholdCandidates = false;
        //
        if (argv[k] == "--on-ic") runComputeIterativeCandidates = true;
        if (argv[k] == "--off-ic") runComputeIterativeCandidates = false;
        //
        if (argv[k] == "--on-cg") runBuildCandidateGraph = true;
        if (argv[k] == "--off-cg") runBuildCandidateGraph = false;
        //
        if (argv[k] == "--on-cl") runPerformClustering = true;
        if (argv[k] == "--off-cl") runPerformClustering = false;
        //
        if (argv[k] == "--on-el") runExtractLines = true;
        if (argv[k] == "--off-el") runExtractLines = false;
        //
        if (argv[k] == "--on-show") show = true;
        if (argv[k] == "--off-show") show = false;
    }

    //
    std::string prefix1 = "";
    if (beams) prefix1 += "bm_";
    else prefix1 += "st_";
    //
    std::string prefix2 = prefix1;
    if (iterative) prefix2 += "it_";
    else prefix2 += "th_";

    //
    std::string preprocessedName = prefix1 + "preprocessed";
    std::string scoresName = prefix1 + "scores";
    std::string directionsName = prefix1 + "directions";
    std::string thresholdCandidatesName = prefix1 + "th_candidates";
    std::string iterativeCandidatesName = prefix1 + "it_candidates";
    //
    std::string candidatesName = prefix2 + "candidates";
    std::string cgraphName = prefix2 + "cgraph";
    std::string clustersName = prefix2 + "clusters";
    std::string linesName = prefix2 + "lines";

    /// 1: load the image, convert to LAB, scale down if needed
    if (runLoadPreprocessImage) {
        lsd::loadPreprocessImage(
            "original", 
            preprocessedName, 
            "params", 
            beams
        );
    }

    /// 2: compute threshold and iterative candidates (using structure tensor score function or beam score function)
    if (runComputeThresholdCandidates) {
        lsd::computeThresholdCandidates(
            preprocessedName, 
            scoresName, 
            directionsName, 
            thresholdCandidatesName, 
            beams); 
    }
    if (runComputeIterativeCandidates) {
        lsd::computeIterativeCandidates(
            preprocessedName, 
            thresholdCandidatesName,
            iterativeCandidatesName,
            beams
        );
    }

    /// 3: build the candidate graph and group the candidates by line segments, extract and reconstruct the line segments
    if (runBuildCandidateGraph) {
        lsd::buildCandidateGraph(candidatesName, cgraphName);
    }
    if (runPerformClustering) {
        lsd::performClustering(cgraphName, clustersName);
    }
    if (runExtractLines) {
        lsd::extractLines("params", candidatesName, clustersName, linesName);
    }
    
    /// extra
    if (show) {
        lsd::buildShowStateImages(
            "original", "original",
            "params", "preprocessed", 
            "palette",
            "scores", "directions", "score-direction", "t_candidates",
            "candidates", "i_candidates",
            "candidates", "cgraph", "cgraph",
            "clusters", "clustering",
            "lines", "result", "original-lines"
        );
    }

    return 0;
}


