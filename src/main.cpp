#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdexcept> // for standard exception types
#include <cmath>
#include <vector>

const double PI = std::acos(-1.0);

// Configuration start
const int DIRECTIONS = 8;
const double R = 3; // radius for the interesting pixels
const double T = 1; // edge thickness
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

double computeScore(const cv::Mat &F, int yPixel, int xPixel, double unitNormY, double unitNormX) {
    const int R_LOWER = std::floor(R);
    // consider only the pixels s.t. their pixel center (y, x) is on/in the circle with radius R
    int count1 = 0, count2 = 0;
    int r1, g1, b1, r2, b2, g2;
    r1 = g1 = b1 = r2 = b2 = g2 = 0;
    for (int y = std::max(0, yPixel - R_LOWER); y <= std::min(F.rows, yPixel + R_LOWER); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = std::ceil((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = std::floor((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = std::max(0, x1); x <= std::min(F.cols, x2); x++) {
            if (!((x - xPixel)*(x - xPixel) + (y - yPixel)*(y - yPixel) <= R*R)) {
                throw "Assertion failed: pixel outside the cirle!!!";
            }
            // skip the pixels on the line
            int dy = (y - yPixel), dx = (x - xPixel);
            double signedDist = dy*unitNormY + dx*unitNormX;
            if (abs(signedDist) <= T/2) continue; // too close to the line
            // add pixel to the corresponding half-circle
            const cv::Vec3b pixel = F.at<cv::Vec3b>(y, x);
            if (signedDist >= 0) {
                count1++; // the half-circle of the normal vector
                b1 += pixel[0];
                g1 += pixel[1];
                r1 += pixel[2];
            } else {
                count2++; // the half-circle opposite to the normal vector
                b2 += pixel[0];
                g2 += pixel[1];
                r2 += pixel[2];
            }
        }
    }
    // compute the score
    double area1 = 1e-6, area2 = 1e-6; // very small 
    if (count1) area1 += (r1 + g1 + b1) / count1;
    if (count2) area2 += (r2 + g2 + b2) / count2;
    return 1.0 - std::min(area1/area2, area2/area1);
}

std::vector<cv::Mat> computeAllScores(const cv::Mat &F) {
    std::vector<cv::Mat> S;
    // Compute the score matrix for every direction
    for (int d = 0; d < DIRECTIONS; d++) {
        cv::Mat SCORE(F.size(), CV_64F);
        // fix the normal unit vector for the direction
        double rad = d * (PI/DIRECTIONS);
        double unitY = std::sin(rad);
        double unitX = std::cos(rad);
        // compute the score matrix for the direction
        for (int y = 0; y < F.rows; y++) {
            for (int x = 0; x < F.cols; x++) {
                SCORE.at<double>(y, x) = computeScore(F, y, x, unitY, unitX);
            }
        }
        S.emplace_back(SCORE);
    }
    return S;
}


int main() {
    
    // Load an RGB image
    cv::Mat F = cv::imread("../images/table.png", cv::IMREAD_COLOR);
    
    // compute the score for every pixel and every direction
    std::vector<cv::Mat> S = computeAllScores(F);
    
    showImage(F);
    for (auto &directionScore : S) {
        showMatrix(directionScore);
    }

    return 0;
}