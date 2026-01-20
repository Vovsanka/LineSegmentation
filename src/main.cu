#include <iostream>

#include <string>
#include <opencv2/opencv.hpp>

#include "operations.hpp"
#include "score.hpp"
#include "candidate.hpp"
#include "iterative.hpp"
#include "cost.hpp"
#include "working_state.hpp"

void checkGPU();
void loadPreprocessImage(std::string path);
void computeThresholdCandidates();
void computeIterativeCandidates();


int main() {

    checkGPU();

    // loadPreprocessImage("../images/black.png");
    loadPreprocessImage("../images/table.png");
    // loadPreprocessImage("../images/apb1.png");
    // loadPreprocessImage("../images/apb2.png");
    // loadPreprocessImage("../images/apb3.png");

    computeThresholdCandidates();
    
    computeIterativeCandidates();
    
    return 0;
}

void checkGPU() {
    // Check the CUDA devices (GPU support)
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;
    if (!cudaCount) std::exit(1);
}

void loadPreprocessImage(std::string path) {
    // Load an RGB image
    cv::Mat originalF = cv::imread(path, cv::IMREAD_COLOR);
    if (originalF.empty()) std::exit(1);
    showImage(originalF);
    
    // preprocessed original
    cv::Mat cpuF;

    // Convert the image to the LAB color space
    cpuF = convertBGRtoLab(originalF);

    // apply the noise filtering to LAB-image (preserves the edge perception)
    cpuF = filterNoise(cpuF);

    // resize the image (to the reasonable processing size)
    double scale = computeScale(cpuF);
    cpuF = resize(cpuF, scale);
    showImage(cpuF);
    
    // print image size
    std::cout << "Original image size: " << originalF.cols << "x" << originalF.rows << std::endl; 
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << "cpu image size: " << cpuF.cols << "x" << cpuF.rows << std::endl;    

    // save the working state
    saveMatrix(originalF, "original");
    saveMatrix(cpuF, "preprocessed");
}

void computeThresholdCandidates() {
    // load the working state
    cv::Mat cpuF = loadMatrix("preprocessed");

    // Upload the preprocessed matrix to GPU
    cv::cuda::GpuMat F = uploadToGPU(cpuF);

    // compute the best scores for every pixel
    cv::cuda::GpuMat S(F.size(), CV_64F);
    cv::cuda::GpuMat D(F.size(), CV_32S);
    computeBestPixelScores(F, S, D);
    showMatrix(S);
    
    // download the matrices to CPU
    cv::Mat cpuS = downloadToCPU(S);
    cv::Mat cpuD = downloadToCPU(D);
    

    // threshold candidates
    std::vector<Cand> tCandidates = extractSortedThresholdCandidates(cpuS, cpuD);
    showScoreDirectionMatrix(cpuS, cpuD, tCandidates);

    // save the working state
    saveMatrix(cpuS, "scores");
    saveMatrix(cpuD, "directions");
    saveCandidates(tCandidates, "tcandidates");
}

void computeIterativeCandidates() {
    // load the working state
    cv::Mat cpuF = loadMatrix("preprocessed");
    cv::Mat cpuS = loadMatrix("scores");
    cv::Mat cpuD = loadMatrix("directions");
    std::vector<Cand> tCandidates = loadCandidates("tcandidates");

    // iterative search candidates
    std::vector<Cand> candidates = candidateIterativeSearch(
        cpuF.ptr<uchar>(), cpuF.step,
        tCandidates,
        cpuF.cols, cpuF.rows
    );
    showScoreDirectionMatrix(cpuS, cpuD, candidates);

    // save the working state
    saveCandidates(candidates, "candidates");
}