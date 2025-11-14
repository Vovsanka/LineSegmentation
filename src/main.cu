#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <thrust/pair.h>

#include <iostream>
#include <cmath>
#include <vector>


// Configuration start
const int DIRECTIONS = 360;
const double R = 1.5; // radius for the interesting pixels
const double OFFSET = 40; // pixel offset to reduce the noise
const double THICKNESS = 0.2; // line thickness
const double CAND_SCORE = 0.95; // candidate score threshold
// Configuration end

void showMatrix(const cv::Mat &F) {
    cv::Mat fNorm;
    cv::normalize(F, fNorm, 0, 255, cv::NORM_MINMAX);
    fNorm.convertTo(fNorm, CV_8U);
    cv::imshow("Matrix", fNorm);
    cv::waitKey(0);
}

void showImage(const cv::Mat &F) {
    // Show the image
    cv::imshow("Image", F);
    cv::waitKey(0);
}

__host__ __device__ thrust::pair<double,double> directionNormalUnitVector(int d) {
    const double PI = acos(-1.0);
    double rad = d * (PI / DIRECTIONS);
    return thrust::make_pair(sin(rad), cos(rad));
}

__host__ __device__ double computeScore(const uchar* F,
                                        double yPixel, double xPixel,
                                        double unitNormY, double unitNormX,
                                        int width, int height) {
    // consider only the pixels s.t. their pixel center (y, x) is on/in the circle with radius R
    int r1, g1, b1, r2, b2, g2;
    r1 = g1 = b1 = r2 = b2 = g2 = 0;
    int minR, minG, minB;
    minR = minG = minB = 255;
    for (int y = max(0, (int)ceilf(yPixel - R)); y <= min(height, (int)floorf(yPixel + R)); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = ceil((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = floor((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = max(0, x1); x <= min(width, x2); x++) {
            int dy = (y - yPixel), dx = (x - xPixel);
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = abs(signedDist);
            // skip the pixels on the line
            if (dist < THICKNESS/2) continue;
            // add pixel to the corresponding half-circle
            int idx = (y * width + x) * 3;
            int b = (int)F[idx];
            int g = (int)F[idx + 1];
            int r = (int)F[idx + 2];
            int w = R*R - (dx*dx + dy*dy);
            minR = min(minR, r);
            minG = min(minG, g);
            minB = min(minB, b);
            if (signedDist >= 0) { // the half-circle of the normal vector
                b1 += w*b;
                g1 += w*g;
                r1 += w*r;
            } else { // the half-circle opposite to the normal vector
                b2 += w*b;
                g2 += w*g;
                r2 += w*r;
            }
        }
    }
    // equalize intensive and non-intensive colors
    for (int y = max(0, (int)ceilf(yPixel - R)); y <= min(height, (int)floorf(yPixel + R)); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = ceilf((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = floorf((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = max(0, x1); x <= min(width, x2); x++) {
            int dy = (y - yPixel), dx = (x - xPixel);
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = abs(signedDist);
            // skip the pixels on the line
            if (dist < 0.1) continue;
            // add pixel to the corresponding half-circle
            int w = R*R - (dx*dx + dy*dy);
            if (signedDist >= 0) {
                b1 += w*(-minB);
                g1 += w*(-minG);
                r1 += w*(-minR);
            } else { // the half-circle opposite to the normal vector
                b2 += w*(-minB);
                g2 += w*(-minG);
                r2 += w*(-minR);
            }
        }
    }
    // compute the score (add offset to reduce the noise sensitivity)
    double area1 = r1 + g1 + b1 + OFFSET;
    double area2 = r2 + g2 + b2 + OFFSET;
    double ratio = max(area1/area2, area2/area1); // avoid div!
    double sqrRatio = ratio*ratio;
    return 1.0 - 1/(sqrRatio*sqrRatio);
}

__global__ void bestScoreKernel(const uchar* F, double* S, int* D,
                                int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    // Compute the score matrix for every direction
    double bestScore = -1;
    double bestDir = 0;

    for (int d = 0; d < DIRECTIONS; ++d) {
        auto [unitY, unitX] = directionNormalUnitVector(d);
        double score = computeScore(F, y, x, unitX, unitY, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }

    int idx = y * width + x;
    S[idx] = bestScore;
    D[idx] = bestDir;
}

__global__ void candidateThresholdKernel(const double *S, const uchar *D, uchar *C,
                                         int width, int height) {
                                    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    double score = S[idx];
    C[idx] = (score >= CAND_SCORE) ? 1 : 0;
}

__host__ std::pair<double,double> findFractionalCandidate(const cv::Mat &F,
                                                        int y, int x, int d) {
    auto [unitY, unitX] = directionNormalUnitVector(d);
    const double step = 0.1;
    // TODO: iterate over fractional coordinates (-1, + 1) in both x- and y-direction
    return std::make_pair(y, x);
}

__host__ std::pair<cv::Mat, cv::Mat> candidateIterativeSearch(const cv::Mat &F,
                                                              const cv::Mat &S,
                                                              const cv::Mat &D) {
    // sort the threshold candidates
    std::vector<std::tuple<double, int, int>> bestPixels; 
    for (int y = 0; y < F.rows; y++) {
        for (int x = 0; x < F.cols; x++) {
            double score = S.at<double>(y, x);
            if (score >= CAND_SCORE) {
                bestPixels.push_back(std::make_tuple(-score, y, x)); // - to sort descending
            }
        }
    }
    std::sort(std::begin(bestPixels), std::end(bestPixels));
    // compute the candidate pixels and directions
    cv::Mat CI(F.size(), CV_64F);
    cv::Mat DI(F.size(), CV_32S);
    for (auto& [score, y, x] : bestPixels) {
        int d = D.at<int>(y, x);
        auto [startY, startX] = findFractionalCandidate(F, y, x, d);
        // TODO: iterative search from the fractional candidate
    }
    return std::make_pair(CI, DI);
}


int main() {
    
    // Check the CUDA devices
    int cudaCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << cudaCount << std::endl;

    // Load an RGB image
    cv::Mat cpuF = cv::imread("../images/table.png", cv::IMREAD_COLOR);
    
    // Upload the image to GPU
    cv::cuda::GpuMat F;
    F.upload(cpuF);

    // GPU threads for each pixel
    dim3 block(16, 16); // 256
    dim3 grid((F.cols + block.x - 1) / block.x, (F.rows + block.y - 1) / block.y); // round up to cover the whole image
    
    // compute the best scores for every pixel
    cv::cuda::GpuMat S(F.size(), CV_64F);
    cv::cuda::GpuMat D(F.size(), CV_32S);
    bestScoreKernel<<<grid, block>>>(
        F.ptr<uchar>(), S.ptr<double>(), D.ptr<int>(),
        F.cols, F.rows
    );

    // download the matrices to CPU
    cv::Mat cpuS, cpuD;
    S.download(cpuS);
    D.download(cpuD);

    // choose the candidates
    cv::cuda::GpuMat C(F.size(), CV_8U);
    candidateThresholdKernel<<<grid, block>>>(
        S.ptr<double>(), D.ptr<uchar>(), C.ptr<uchar>(),
        F.cols, F.rows
    );

    // download the matrices to CPU
    cv::Mat cpuC;    
    C.download(cpuC);

    // show the images 
    showImage(cpuF);
    showMatrix(cpuS);
    showMatrix(cpuC);

    // // choose the candiates upgraded
    // cv::Mat cpuCI, cpuDI;
    // std::tie(cpuCI, cpuDI) = candidateIterativeSearch(cpuF, cpuS, cpuD);

    // showMatrix(cpuCI);

    return 0;
}