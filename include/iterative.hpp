#ifndef ITERATIVE_HPP
#define ITERATIVE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#include "config.hpp"
#include "score.hpp"

__host__
std::pair<double,double> findFractionalCandidate(const cv::Mat &F,
                                                 int y, int x, int d);

__host__
std::pair<cv::Mat, cv::Mat> candidateIterativeSearch(const cv::Mat &F,
                                                     const cv::Mat &S,
                                                     const cv::Mat &D);

#endif