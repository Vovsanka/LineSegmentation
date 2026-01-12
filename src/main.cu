#include <iostream>

#include <opencv2/opencv.hpp>

#include "operations.hpp"
#include "score.hpp"
#include "candidate.hpp"
#include "iterative.hpp"



int main() {

    // Check the CUDA devices (GPU support)
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;
    if (!cudaCount) return 1;

    // Load an RGB image
    // cv::Mat originalF = cv::imread("../images/black.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/mini-table.png", cv::IMREAD_COLOR);
    cv::Mat originalF = cv::imread("../images/table.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb1.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb2.png", cv::IMREAD_COLOR);
    // cv::Mat originalF = cv::imread("../images/apb3.png", cv::IMREAD_COLOR);
    if (originalF.empty()) return 1;
    std::cout << "Original image size: " << originalF.cols << "x" << originalF.rows << std::endl; 
    showImage(originalF);
    
    // Convert the image to the LAB color space
    cv::Mat labF = convertBGRtoLab(originalF);
    // showImage(labF);

    // apply the noise filtering to LAB-image (preserves the edge perception)
    cv::Mat filteredF = filterNoise(labF);
    // showImage(filteredF);

    // resize the image (to the reasonable processing size)
    double scale = computeScale(filteredF);
    cv::Mat scaledF = resize(filteredF, scale);
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << "Scaled image size: " << scaledF.cols << "x" << scaledF.rows << std::endl; 
    // showImage(scaledF);
    
    // Upload the image to GPU
    cv::cuda::GpuMat F = uploadToGPU(scaledF);
    // showImage(F);

    // compute the best scores for every pixel
    cv::cuda::GpuMat S(F.size(), CV_64F);
    cv::cuda::GpuMat D(F.size(), CV_32S);
    computeBestPixelScores(F, S, D);
    // showMatrix(S);
    
    // download the matrices to CPU
    cv::Mat Fcpu = downloadToCPU(F);
    cv::Mat Scpu = downloadToCPU(S);
    cv::Mat Dcpu = downloadToCPU(D);
    
    // threshold candidates
    std::vector<Cand> tCandidates = extractSortedThresholdCandidates(Scpu, Dcpu);
    showScoreDirectionMatrix(Scpu, Dcpu, tCandidates);
    
    // iterative search candidates
    std::vector<Cand> candidates = candidateIterativeSearch(
        Fcpu.ptr<uchar>(), Fcpu.step,
        tCandidates,
        F.cols, F.rows
    );
    showScoreDirectionMatrix(Scpu, Dcpu, candidates);

    return 0;
}