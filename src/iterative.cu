// #include "iterative.hpp"


// __host__ __device__
// thrust::tuple<float,float,float> upgradeCandidate(const uchar* F,
//                                                      thrust::tuple<float,float,float> cand,
//                                                      int width, int height) {
//     float yPixel = thrust::get<0>(cand);
//     float xPixel = thrust::get<1>(cand);
//     float dirRad = thrust::get<2>(cand);
//     //
//     thrust::tuple<float,float> unitNorm = getUnitVector(dirRad);
//     float unitNormY = thrust::get<0>(unitNorm);
//     float unitNormX = thrust::get<1>(unitNorm);
//     //
//     float bestScore = -1;
//     float bestDir = 0;
//     float bestY = 0, bestX = 0;
//     for (int k = -UP_COUNT; k <= UP_COUNT; k++) {
//             float y = yPixel + k*UP_STEP*unitNormY;
//             float x = xPixel + k*UP_STEP*unitNormX;
//             thrust::tuple<float,float> newScoreDir = bestPossibleScore(F, y, x, width, height);
//             float newScore = thrust::get<0>(newScoreDir);
//             float newDir = thrust::get<1>(newScoreDir);
//             if (newScore > bestScore) {
//                 bestScore = newScore;
//                 bestDir =  newDir;
//                 bestY = y;
//                 bestX = x;

//             }
//     }
//     //
//     return thrust::make_tuple(bestY, bestX, bestDir);
// }

// __host__
// std::vector<std::tuple<float,float,float>> sortThresholdCandidates(const float *S, int width, int height) {
//     std::vector<std::tuple<float,float,float>> tCand;
//     for (int y = 0; y < height; y++) {
//         for (int x = 0; x < width; x++) {
//             int idx = y * width + x;
//             float score = S[idx];
//             if (score >= START_THRESHOLD) {
//                 tCand.push_back(std::make_tuple(-score, y, x));
//             }
//         }
//     }
//     std::sort(std::begin(tCand), std::end(tCand));
//     return tCand;
// }

// __host__
// cv::Mat candidateIterativeSearch(const uchar* F, const float *S, const int *D, int width, int height) {
//     std::vector<std::tuple<float,float,float>> tCand = sortThresholdCandidates(S, width, height);
//     cv::Mat CI(height, width, CV_8U, cv::Scalar(0));
//     for (std::tuple<float,float,float> start: tCand) {
//         float startY = std::get<1>(start);
//         float startX = std::get<2>(start);
//         int idx = startY * width + startX;
//         float direction = getRad(D[idx]);
//         thrust::tuple<float,float,float> cand = thrust::make_tuple(startY, startX, direction);
//         for (int k = 0; k < UP_ITERATIONS; k++) {
//             cand = upgradeCandidate(F, cand, width, height);
//         }
//         //
//         startY = thrust::get<0>(cand);
//         startX = thrust::get<1>(cand);
//         direction = thrust::get<2>(cand);
//         candidateExpand(F, CI.ptr(), startY, startX, direction, width, height); 
//         //
//         showMatrix(CI);
//     }
//     return CI;
// }


// __host__
// void candidateExpand(
//     const uchar *F, uchar *CI, 
//     float startY, float startX,
//     float dirRad, 
//     int width, int height) {
//     //
//     thrust::tuple<float,float> unitEdge = getUnitVector(dirRad + getPi()/2.0);
//     float unitEdgeY = thrust::get<0>(unitEdge);
//     float unitEdgeX = thrust::get<1>(unitEdge);
//     //
//     for (int k = 0; ; k++) {
//         float y1 = startY + k*unitEdgeY;
//         float x1 = startX + k*unitEdgeX;
//         float y2 = startY - k*unitEdgeY;
//         float x2 = startX - k*unitEdgeX;
//         if (
//             !setCandidates(F, CI, y1, x1, dirRad, width, height)
//             &&
//             !setCandidates(F, CI, y2, x2, dirRad, width, height)
//         ) break;
//     }
//     return;
// }


// __host__ 
// bool setCandidates(
//     const uchar *F, uchar *CI, 
//     float y, float x,
//     float dirRad, 
//     int width, int height) {
//     //
//     if (
//         y < 0 || y >= height ||
//         x < 0 || x >= width
//     ) return false;
//     //
//     int downY = floor(y);
//     int downX = floor(x);
//     int upY = ceil(y);
//     int upX = ceil(x);
//     //
//     bool isSet = false;
//     //
//     if (downY >= 0 && downX >= 0) {
//         float score = computeLabScore(F, downY, downX, dirRad, width, height);
//         if (score >= MIN_THRESHOLD) {
//             int idx = downY*width + downX;
//             CI[idx] = 1;
//             isSet = true;
//         }
//     }
//     //
//     if (downY >= 0 && upX < width) {
//         float score = computeLabScore(F, downY, upX, dirRad, width, height);
//         if (score >= THRESHOLD) {
//             int idx = downY*width + upX;
//             CI[idx] = 1;
//             isSet = true;
//         }
//     }
//     //
//     if (upY < height && downX >= 0) {
//         float score = computeLabScore(F, upY, downX, dirRad, width, height);
//         if (score >= MIN_THRESHOLD) {
//             int idx = upY*width + downX;
//             CI[idx] = 1;
//             isSet = true;
//         }
//     }
//     //
//     if (upY < height && upX < width) {
//         float score = computeLabScore(F, upY, upX, dirRad, width, height);
//         if (score >= MIN_THRESHOLD) {
//             int idx = upY*width + upX;
//             CI[idx] = 1;
//             isSet = true;
//         }
//     }
//     //
//     if (isSet) {
//         std::cout << "Set candidates for: y=" << y << " x=" << x  << std::endl;
//     }
//     return isSet;
// }