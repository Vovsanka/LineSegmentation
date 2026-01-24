#ifndef WORKING_STATE_HPP
#define WORKING_STATE_HPP

#include <string>
#include <fstream>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "cand_type.hpp"
#include "cgraph_type.hpp"
#include "line_type.hpp"


const std::filesystem::path pathPrefix("../working-state");

void saveMatrix(const cv::Mat& M, std::string name);

cv::Mat loadMatrix(std::string name);

void saveCandidates(const std::vector<Cand>& candidates, std::string name);

std::vector<Cand> loadCandidates(std::string name);

void saveCandidateGraph(const CandidateGraph& G, std::string name);

CandidateGraph loadCandidateGraph(std::string name);

void saveEdgeLabels(const std::vector<char>& edgeLabels, std::string name);

std::vector<char> loadEdgeLabels(std::string name);

void saveLines(const std::vector<Line>& lines, std::string name);

std::vector<Line> loadLines(std::string name);

#endif