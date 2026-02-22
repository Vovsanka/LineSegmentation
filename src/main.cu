#include <iostream>

#include <string>
#include <opencv2/opencv.hpp>

#include "operations.hpp"
#include "candidate.hpp"
#include "iterative.hpp"
#include "clustering.hpp"
#include "lines.hpp"


// independent steps 
void checkGPU(); // 0
void loadPreprocessImage(
    std::string originalImage_outName,
    std::string preprocessedImage_outName,
    std::string params_outName,
    bool grayscale = false
); // 1
void computeThresholdCandidates(
    std::string preprocessedImage_inName,
    std::string scoreMatrix_outName,
    std::string directionMatrix_outName,
    std::string candidateList_outName,
    bool beamScore = true
); // 2-1
void computeIterativeCandidates(
    std::string preprocessedImage_inName,
    std::string candidateList_inName,
    std::string candidateList_outName
); // 2-2
void showCandidates(
    std::string scoreMatrix_inName,
    std::string directionMatrix_inName,
    std::string candidateList_inName
); // 2.1
void buildCandidateGraph(
    std::string candidateList_inName,
    std::string candidateGraph_outName
); // 3
void performClustering(
    std::string candidateGraph_inName,
    std::string edgeLabels_outName
); // 4
void buildClusterImage(
    std::string params_inName,
    std::string candidateList_inName,
    std::string candidateGraph_inName,
    std::string edgeLabels_inName,
    std::string clusterImage_outName
); // 4.1
void extractLines(
    std::string candidateList_inName,
    std::string candidateGraph_inName,
    std::string edgeLabels_inName,
    std::string lines_outName
); // 5
void reconstructOriginalLines(
    std::string params_inName,
    std::string lines_inName,
    std::string lines_outName
); // 5.1
void buildLineEdgeImage(
    std::string params_inName,
    std::string lines_inName,
    std::string lineEdgeImage_outName,
    bool originalSize = false
); // 5.2


int main() {

    checkGPU();

    loadPreprocessImage("original", "preprocessed", "params");

    computeThresholdCandidates("preprocessed", "scores", "directions", "t_candidates");
    showCandidates("scores", "directions", "t_candidates");
    
    computeIterativeCandidates("preprocessed", "t_candidates", "candidates");
    showCandidates("scores", "directions", "candidates");

    buildCandidateGraph("candidates", "cgraph");
    performClustering("cgraph", "labels");
    buildClusterImage("params", "candidates", "cgraph", "labels", "clusters");

    extractLines("candidates", "cgraph", "labels", "lines");
    buildLineEdgeImage("params", "lines", "edges");

    reconstructOriginalLines("params", "lines", "or_lines");
    buildLineEdgeImage("params", "or_lines", "or_edges", true);

    return 0;
}

void checkGPU() {
    // Check the CUDA devices (GPU support)
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;
    if (!cudaCount) std::exit(1);
}

void loadPreprocessImage(
    std::string originalImage_outName,
    std::string preprocessedImage_outName,
    std::string params_outName,
    bool grayscale
) {
    // Load an RGB image
    cv::Mat originalF = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
    if (originalF.empty()) std::exit(1);
    showImage(originalF);
    
    // preprocessed original
    cv::Mat cpuF;

    // Convert the image to the LAB color space
    if (grayscale) {
        cpuF = convertBGRtoGrayscale(originalF);
    } else {
        cpuF = convertBGRtoLab(originalF);
    }

    // resize the image (to the reasonable processing size)
    double scale = computeScale(cpuF);
    cpuF = resize(cpuF, scale);
    showImage(cpuF);
    
    // print image size
    std::cout << "Original image size: " << originalF.cols << "x" << originalF.rows << std::endl; 
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << "cpu image size: " << cpuF.cols << "x" << cpuF.rows << std::endl;    

    // save the working state
    saveMatrix(originalF, originalImage_outName);
    saveMatrix(cpuF, preprocessedImage_outName);
    saveImageParams(originalF.cols, originalF.rows, cpuF.cols, cpuF.rows, params_outName);
}

void computeThresholdCandidates(
    std::string preprocessedImage_inName,
    std::string scoreMatrix_outName,
    std::string directionMatrix_outName,
    std::string candidateList_outName,
    bool beamScore
) {
    // load the working state
    cv::Mat cpuF = loadMatrix(preprocessedImage_inName);

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

    // save the working state
    saveMatrix(cpuS, scoreMatrix_outName);
    saveMatrix(cpuD, directionMatrix_outName);
    saveCandidates(tCandidates, candidateList_outName);
}

void computeIterativeCandidates(
    std::string preprocessedImage_inName,
    std::string candidateList_inName,
    std::string candidateList_outName
) {
    // load the working state
    cv::Mat cpuF = loadMatrix(preprocessedImage_inName);
    std::vector<Cand> tCandidates = loadCandidates(candidateList_inName);

    // iterative search candidates
    cv::cuda::GpuMat gpuF = uploadToGPU(cpuF);
    std::vector<Cand> candidates = candidateIterativeSearch(
        cpuF.ptr<uchar>(), cpuF.step,
        gpuF,
        tCandidates,
        cpuF.cols, cpuF.rows
    );

    // save the working state
    saveCandidates(candidates, candidateList_outName);
}

void showCandidates(
    std::string scoreMatrix_inName,
    std::string directionMatrix_inName,
    std::string candidateList_inName
) {
    // load the working state
    cv::Mat cpuS = loadMatrix(scoreMatrix_inName);
    cv::Mat cpuD = loadMatrix(directionMatrix_inName);
    std::vector<Cand> candidates = loadCandidates(candidateList_inName);
    //
    showScoreDirectionMatrix(cpuS, cpuD, candidates);
}

void buildCandidateGraph(
    std::string candidateList_inName,
    std::string candidateGraph_outName
) {
    // load the working state
    std::vector<Cand> candidates = loadCandidates(candidateList_inName);

    CandidateGraph G(candidates);

    // save the working state
    saveCandidateGraph(G, candidateGraph_outName);
}

void performClustering(
    std::string candidateGraph_inName,
    std::string edgeLabels_outName
) {
    // load the working state
    CandidateGraph G = loadCandidateGraph(candidateGraph_inName);

    std::vector<char> edgeLabels = solveClustering(G);

    // save the working state
    saveEdgeLabels(edgeLabels, edgeLabels_outName);
}

void buildClusterImage(
    std::string params_inName,
    std::string candidateList_inName,
    std::string candidateGraph_inName,
    std::string edgeLabels_inName,
    std::string clusterImage_outName
) {
    // load the working state
    int originalWidth, originalHeight;
    int width, height;
    std::tie(originalWidth, originalHeight, width, height) = loadImageParams(params_inName);
    std::vector<Cand> candidates = loadCandidates(candidateList_inName);
    CandidateGraph G = loadCandidateGraph(candidateGraph_inName);
    std::vector<char> edgeLabels = loadEdgeLabels(edgeLabels_inName);

    // draw the clusters as connected candidates
    drawClusterImage(width, height, candidates, G, edgeLabels, clusterImage_outName);

     // show the line edge image
    cv::Mat I = cv::imread(
        (WORKING_STATE_DIR/(clusterImage_outName + ".png")).string().c_str(),
        cv::IMREAD_GRAYSCALE
    );
    showImage(I);
}

void extractLines(
    std::string candidateList_inName,
    std::string candidateGraph_inName,
    std::string edgeLabels_inName,
    std::string lines_outName
) {
    // load the working state
    std::vector<Cand> candidates = loadCandidates(candidateList_inName);
    CandidateGraph G = loadCandidateGraph(candidateGraph_inName);
    std::vector<char> edgeLabels = loadEdgeLabels(edgeLabels_inName);

    std::vector<Line> lines = extractLinesFromClusters(candidates, G, edgeLabels);

    // save the working state
    saveLines(lines, lines_outName);
}

void reconstructOriginalLines(
    std::string params_inName,
    std::string lines_inName,
    std::string lines_outName
) {
    // load the working state
    int originalWidth, originalHeight;
    int width, height;
    std::tie(originalWidth, originalHeight, width, height) = loadImageParams(params_inName);
    std::vector<Line> lines = loadLines(lines_inName);
    
    double scaleY = 1.0*originalHeight/height;
    double scaleX = 1.0*originalWidth/width;
    std::vector<Line> originalLines(lines.size());
    for (int k = 0; k < lines.size(); k++) {
        const Line& l = lines[k];
        originalLines[k] = Line(scaleY*l.y1, scaleX*l.x1, scaleY*l.y2, scaleX*l.x2);
    } 

    // save the working state
    saveLines(originalLines, lines_outName);
}

void buildLineEdgeImage(
    std::string params_inName,
    std::string lines_inName,
    std::string lineEdgeImage_outName,
    bool originalSize
) {
    // load the working state
    int originalWidth, originalHeight;
    int width, height;
    std::tie(originalWidth, originalHeight, width, height) = loadImageParams(params_inName);
    std::vector<Line> lines = loadLines(lines_inName);

    if (originalSize) {
        width = originalWidth;
        height = originalHeight;
    }

    // draw and save the line edge image
    drawLineEdgeImage(lines, width, height, lineEdgeImage_outName);

    // show the line edge image
    cv::Mat I = cv::imread(
        (WORKING_STATE_DIR/(lineEdgeImage_outName + ".png")).string().c_str(),
        cv::IMREAD_GRAYSCALE
    );
    showImage(I);
}
