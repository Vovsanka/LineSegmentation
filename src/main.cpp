#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>
#include <vector>

const double PI = std::acos(-1.0);

// Configuration start
const int SOBEL_KSIZE = 3; // kernel size of the sobel filter
const double SIGMA = 1.0; // standard deviation of the Gaussian
const cv::Size NEIGHBORHOOD = cv::Size(3, 3); // neighborhood for the Gausssian smoothing
const double THRESHOLD_EIGENVALUE = 0.25; // *mean_eigenvalue1 (determine large/small)
const int DIRECTIONS = 8;
// Configuration end

void showImage(const cv::Mat &F) {
    cv::Mat fNorm;
    cv::normalize(F, fNorm, 0, 255, cv::NORM_MINMAX);
    fNorm.convertTo(fNorm, CV_8U);
    cv::imshow("Matrix", fNorm);
    cv::waitKey(0);
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> computeStructureTensorComponents(const cv::Mat &F) {
    // Compute image gradients (Sobel filter)
    cv::Mat Ix, Iy;
    cv::Sobel(F, Ix, CV_64F, 1, 0, SOBEL_KSIZE); // dI/dx
    cv::Sobel(F, Iy, CV_64F, 0, 1, SOBEL_KSIZE); // dI/dy
    // Compute structure tensor components
    cv::Mat Ix2 = Ix.mul(Ix);     // Ix^2
    cv::Mat Iy2 = Iy.mul(Iy);     // Iy^2
    cv::Mat Ixy = Ix.mul(Iy);     // Ix*Iy
    // Apply Gaussian smoothing to component
    cv::Mat Jxx, Jyy, Jxy;
    cv::GaussianBlur(Ix2, Jxx, NEIGHBORHOOD, SIGMA);
    cv::GaussianBlur(Iy2, Jyy, NEIGHBORHOOD, SIGMA);
    cv::GaussianBlur(Ixy, Jxy, NEIGHBORHOOD, SIGMA);
    // return the structure tensor components
    return std::make_tuple(Jxx, Jyy, Jxy);
}

std::tuple<cv::Mat, cv::Mat> computeStructureTensorEigenvalues(const cv::Mat &Jxx, const cv::Mat &Jyy, const cv::Mat &Jxy) {
    // build the structure tensors for each pixel and compute the eigenvalues  
    cv::Mat L1(Jxx.size(), CV_64F);
    cv::Mat L2(Jxx.size(), CV_64F);
    for (int y = 0; y < Jxx.rows; y++) {
        for (int x = 0; x < Jxx.cols; x++) {
            // build the structure tensor for the pixel
            cv::Mat J = (cv::Mat_<double>(2, 2) <<
                Jxx.at<double>(y, x), Jxy.at<double>(y, x),
                Jxy.at<double>(y, x), Jyy.at<double>(y, x)
            );
            // compute eigenvalues and eigenvectors
            cv::Mat eigenvalues, eigenvectors;
            cv::eigen(J, eigenvalues, eigenvectors);
            // Store in lambda maps
            double l1 = eigenvalues.at<double>(0);
            double l2 = eigenvalues.at<double>(1);
            if (l1 < l2) std::swap(l1, l2);
            L1.at<double>(y, x) = l1;
            L2.at<double>(y, x) = l2;
        }
    }
    return std::make_tuple(L1, L2);
}

cv::Mat classifyPixels(const cv::Mat &L1, const cv::Mat &L2) {
    // classify edges and corners
    cv::Mat T(L1.size(), CV_64F);
    cv::Scalar meanL1 = cv::mean(L1);
    double threshold = THRESHOLD_EIGENVALUE * meanL1[0];
    for (int y = 0; y < L1.rows; y++) {
        for (int x = 0; x < L1.cols; x++) {
            double l1 = L1.at<double>(y, x);
            double l2 = L2.at<double>(y, x);
            int pixelType = 0;
            if (l1 > threshold) { // edge or corner
                if (l2 > threshold) {
                    pixelType = 255; // corner
                } else {
                    pixelType = 100; // edge
                }
            }
            T.at<double>(y,x) = pixelType;
        }
    }
    return T;
}

std::vector<cv::Mat> computeAllScores(
    const cv::Mat &Jxx,
    const cv::Mat &Jyy,
    const cv::Mat &Jxy,
    const cv::Mat &L1,
    const cv::Mat &L2,
    const cv::Mat &T
) {
    std::vector<cv::Mat> S;
    // Compute the score matrix for every direction
    for (int d = 0; d < DIRECTIONS; d++) {
        // fix the normal unit vector for the direction
        double rad = d * (PI/DIRECTIONS);
        double unitY = std::asin(rad);
        double unitX = std::acos(rad);
        // compute the score matrix for the direction
        cv::Mat SCORE(Jxx.size(), CV_64F);
        for (int y = 0; y < Jxx.rows; y++) {
            for (int x = 0; x < Jxx.cols; x++) {
                double jxx = Jxx.at<double>(y, x);
                double jyy = Jyy.at<double>(y, x);
                double jxy = Jxy.at<double>(y, x);
                double l1 = L1.at<double>(y, x);
                double l2 = L2.at<double>(y, x);
                double e = T.at<double>(y, x);
                double score = 0;
                if (e > 0) {
                    double pr = unitY*unitY*jxx + 2*unitY*unitX*jxy + unitX*unitX*jyy;
                    score = (pr - l2) / (l1 - l2);
                }
                SCORE.at<double>(y, x) = score;
            }
        }
        S.emplace_back(SCORE);
    }
    return S;
}


int main() {
    
    // Load a grayscale image
    cv::Mat F = cv::imread("../images/table.png", cv::IMREAD_GRAYSCALE);

    // showImage(F);

    // Process the image
    cv::Mat Jxx, Jyy, Jxy;
    std::tie(Jxx, Jyy, Jxy) = computeStructureTensorComponents(F);

    cv::Mat L1, L2;
    std::tie(L1, L2) = computeStructureTensorEigenvalues(Jxx, Jyy, Jxy);

    cv::Mat T = classifyPixels(L1, L2);

    // showImage(T);

    // Compute all scores
    std::vector<cv::Mat> S = computeAllScores(Jxx, Jyy, Jxy, L1, L2, T);

    for (int d = 0; d < DIRECTIONS; d++) {
        showImage(S[d]);
    }

    return 0;
}