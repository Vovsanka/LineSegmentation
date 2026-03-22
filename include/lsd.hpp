#ifndef LSD_HPP
#define LSD_HPP

#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>
#include <random>

#include "config.hpp"
#include "operations.hpp"
#include "candidate.hpp"
#include "iterative.hpp"
#include "clustering.hpp"
#include "lines.hpp"


namespace lsd { // Line Segment Detection

    // filesystem paths
    extern std::filesystem::path imagePath;
    extern std::filesystem::path workingStateDir;

    // independent steps 
    void checkGPU(); // 0

    void loadPreprocessImage( // 1
        std::string originalImage_outName,
        std::string preprocessedImage_outName,
        std::string params_outName,
        bool colorImage = true
    ); 
    
    void computeThresholdCandidates( // 2.1
        std::string preprocessedImage_inName,
        std::string scoreMatrix_outName,
        std::string directionMatrix_outName,
        std::string thresholdCandidates_outName,
        bool beamScore = true
    ); 

    void computeIterativeCandidates( // 2.2
        std::string preprocessedImage_inName,
        std::string thresholdCandidates_inName,
        std::string iterativeCandidates_outName,
        bool beamScore = true
    ); 

    void buildCandidateGraph( // 3
        std::string candidateList_inName,
        std::string candidateGraph_outName
    ); 

    void performClustering( // 3.1
        std::string candidateGraph_inName,
        std::string edgeLabels_outName
    ); 

    void extractLines( // 3.2
        std::string candidateList_inName,
        std::string candidateGraph_inName,
        std::string edgeLabels_inName,
        std::string scaledLines_outName
    ); 

    // void reconstructOriginalLines( // 3.3
    //     std::string params_inName,
    //     std::string scaledLines_inName,
    //     std::string originalLines_outName
    // ); 

    void buildShowStateImages( // extra
        std::string originalImage_inName = "",
        std::string originalImage_outName = "",
        //
        std::string params_inName = "",
        std::string preprocessedImage_inName = "",
        //
        std::string scoreMatrix_inName = "",
        std::string directionMatrix_inName = "",
        std::string scoreDirection_outName = "",
        std::string thresholdCandidates_outName = "",
        //
        std::string iterativeCandidates_inName = "",
        std::string iterativeCandidates_outName = "",
        //
        std::string candidateList_inName = "",
        std::string candidateGraph_inName = "",
        std::string candidateGraph_outName = "",        
        //
        std::string edgeLabels_inName = "",
        std::string clustering_outName = "",
        //
        std::string lines_inName = "",
        std::string lines_outName = "",
        std::string originalLines_outName = ""
    );

    /////////////////

    cv::Mat buildScoreDirectionMatrix(
        cv::Mat& S,
        cv::Mat& D,
        double threshold = 0
    );

    void buildGraphImage(
        std::string& name, 
        int width, int height,
        const std::vector<Cand>& candidates, 
        const CandidateGraph& cgraph = CandidateGraph(),
        const std::vector<char>& edgeLabels = std::vector<char>()
    );

    void buildLineImage(
        std::string& originalLines_outName, 
        int width, int height,
        const std::vector<Line>& lines,
        std::string originalName = ""
    );

    /////////////////

    void saveMatrix(const cv::Mat& M, std::string name);

    cv::Mat loadMatrix(std::string name);

    void saveImageParams(
        int width, int height,
        std::string& name
    );

    void loadImageParams(
        std::string& name,
        int& width, int& height
    );

    void saveCandidates(const std::vector<Cand>& candidates, std::string& name);

    std::vector<Cand> loadCandidates(std::string& name);

    void saveCandidateGraph(const CandidateGraph& G, std::string& name);
    CandidateGraph loadCandidateGraph(std::string& name);

    void saveEdgeLabels(const std::vector<char>& edgeLabels, std::string& name);

    std::vector<char> loadEdgeLabels(std::string& name);

    void saveLines(const std::vector<Line>& lines, std::string& name);

    std::vector<Line> loadLines(std::string& name);

    

};

#endif