#include "iterative.hpp"


__host__ __device__
thrust::tuple<double,double,double> upgradeCandidate(const uchar* F,
                                                     double yPixel, double xPixel,
                                                     double score, double dirRad, 
                                                     int width, int height) {
    thrust::tuple<double,double> unitNorm = getUnitVector(dirRad);
    double unitNormY = thrust::get<0>(unitNorm);
    double unitNormX = thrust::get<1>(unitNorm);
    //
    for (int i = -UP_COUNT; i <= UP_COUNT; i++) {
        for (int j = -UP_COUNT; j <= UP_COUNT; j++) {
            double y = yPixel + i*UP_STEP*unitNormY;
            double x = xPixel + i*UP_STEP*unitNormX;
            thrust::tuple<double,double> newScoreDir = bestPossibleScore(F, y, x, width, height);
            double unitNormY = thrust::get<0>(unitNorm);
            double unitNormX = thrust::get<1>(unitNorm);
        }
    }

    //
    return thrust::make_tuple(0,0,0);
}


// __host__ 
// std::pair<double,double> findFractionalCandidate(const cv::Mat &F,
//                                                         int y, int x, int d) {
//     auto [unitY, unitX] = directionNormalUnitVector(d);
//     const double step = 0.1;
//     // TODO: iterate over fractional coordinates (-1, + 1) in both x- and y-direction
//     return std::make_pair(y, x);
// }

// __host__
// std::pair<cv::Mat, cv::Mat> candidateIterativeSearch(const cv::Mat &F,
//                                                               const cv::Mat &S,
//                                                               const cv::Mat &D) {
//     // sort the threshold candidates
//     std::vector<std::tuple<double, int, int>> bestPixels; 
//     for (int y = 0; y < F.rows; y++) {
//         for (int x = 0; x < F.cols; x++) {
//             double score = S.at<double>(y, x);
//             if (score >= THRESHOLD) {
//                 bestPixels.push_back(std::make_tuple(-score, y, x)); // - to sort descending
//             }
//         }
//     }
//     std::sort(std::begin(bestPixels), std::end(bestPixels));
//     // compute the candidate pixels and directions
//     cv::Mat CI(F.size(), CV_64F);
//     cv::Mat DI(F.size(), CV_32S);
//     for (auto& [score, y, x] : bestPixels) {
//         int d = D.at<int>(y, x);
//         auto [startY, startX] = findFractionalCandidate(F, y, x, d);
//         // TODO: iterative search from the fractional candidate
//     }
//     return std::make_pair(CI, DI);
// }