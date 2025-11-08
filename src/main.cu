#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <vector>


// Configuration start
const int DIRECTIONS = 360;
const double R = 2.5; // radius for the interesting pixels
const double OFFSET = 500; // pixel offset
const double CAND_SCORE = 0.9; // candidate score threshold
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

template <typename T>
__device__ T max(T a, T b) {
    return (a > b) ? a : b;
}

template <typename T>
__device__ T min(T a, T b) {
    return (a > b) ? a : b;
}

__device__ double computeScore(const float* F,
                               int yPixel, int xPixel,
                               double unitNormY, double unitNormX,
                               int width, int height) {
    const int R_LOWER = std::floor(R);
    int idx = (yPixel * width + xPixel) * 3;
    // consider only the pixels s.t. their pixel center (y, x) is on/in the circle with radius R
    double r1, g1, b1, r2, b2, g2;
    r1 = g1 = b1 = r2 = b2 = g2 = 0;
    double minR, minG, minB;
    minR = minG = minB = 255;
    for (int y = max(0, yPixel - R_LOWER); y <= min(height, yPixel + R_LOWER); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = ceil((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = floor((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = max(0, x1); x <= min(width, x2); x++) {
            int dy = (y - yPixel), dx = (x - xPixel);
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = std::abs(signedDist);
            // skip the pixels on the line
            if (dist < 0.1) continue;
            // add pixel to the corresponding half-circle
            double b = F[idx], g = F[idx + 1], r = F[idx + 2];
            double w = R*R - (dx*dx + dy*dy);
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
    for (int y = max(0, yPixel - R_LOWER); y <= min(height, yPixel + R_LOWER); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = std::ceil((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = std::floor((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = max(0, x1); x <= min(width, x2); x++) {
            int dy = (y - yPixel), dx = (x - xPixel);
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = std::abs(signedDist);
            // skip the pixels on the line
            if (dist < 0.1) continue;
            // add pixel to the corresponding half-circle
            double w = R*R - (dx*dx + dy*dy);
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
    double ratio = max(area1/area2, area2/area1);
    return 1.0 - 1/(ratio*ratio*ratio);
}

__global__ void bestScoreKernel(const float* F, double* S, double* D,
                                int width, int height, int directions) {
    const double PI = std::acos(-1.0);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    // Compute the score matrix for every direction
    double bestScore = -1;
    double bestDir = 0;

    for (int d = 0; d < directions; ++d) {
        double rad = d * (PI / directions);
        double unitY = sin(rad);
        double unitX = cos(rad);

        double score = computeScore(F, x, y, unitX, unitY, width, height);
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }

    int idx = y * width + x;
    S[idx] = bestScore;
    D[idx] = bestDir;
}

cv::Mat chooseCandiates(const cv::Mat &S) {
    cv::Mat C(S.size(), CV_64F);
    for (int y = 0; y < S.rows; y++) {
        for (int x = 0; x < S.cols; x++) {
            double score = S.at<double>(y, x);
            C.at<double>(y, x) = (score >= CAND_SCORE) ? 1 : 0;
        }
    }
    return C;
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
    dim3 grid((F.cols + 15) / 16, (F.rows + 15) / 16); // round up to cover the whole image
    
    // compute the best scores for every pixel
    cv::cuda::GpuMat S(F.size(), CV_64F);
    cv::cuda::GpuMat D(F.size(), CV_64F);
    bestScoreKernel<<<grid, block>>>(
        F.ptr<float>(), S.ptr<double>(), D.ptr<double>(),
        F.cols, F.rows, DIRECTIONS
    );

    // // choose the candidates
    // cv::Mat C = chooseCandiates(S);
    
    // download the images to CPU
    cv::Mat cpuS;
    // S.download(cpuS);

    // show the images
    showImage(cpuF);
    // showMatrix(cpuS);
    // showMatrix(cpuC);

    return 0;
}