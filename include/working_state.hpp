#ifndef WORKING_STATE_HPP
#define WORKING_STATE_HPP

#include <string>
#include <fstream>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "cand_type.hpp"
#include "cgraph.hpp"


const std::filesystem::path pathPrefix("../working-state");

void saveMatrix(const cv::Mat& M, std::string name);

cv::Mat loadMatrix(std::string name);

void saveCandidates(const std::vector<Cand>& candidates, std::string name);

std::vector<Cand> loadCandidates(std::string name);

void saveCandidateGraph(const CandidateGraph& G, std::string name);

CandidateGraph loadCandidateGraph(std::string name);

#endif