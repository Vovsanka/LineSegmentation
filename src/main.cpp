#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdexcept> // for standard exception types
#include <cmath>
#include <vector>

const double PI = std::acos(-1.0);

// Configuration start
const int DIRECTIONS = 30;
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

double computeScore(const cv::Mat &F, int yPixel, int xPixel, double unitNormY, double unitNormX) {
    const int R_LOWER = std::floor(R);
    // consider only the pixels s.t. their pixel center (y, x) is on/in the circle with radius R
    double r1, g1, b1, r2, b2, g2;
    r1 = g1 = b1 = r2 = b2 = g2 = 0;
    double minR, minG, minB;
    minR = minG = minB = 255;
    for (int y = std::max(0, yPixel - R_LOWER); y <= std::min(F.rows, yPixel + R_LOWER); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = std::ceil((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = std::floor((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = std::max(0, x1); x <= std::min(F.cols, x2); x++) {
            int dy = (y - yPixel), dx = (x - xPixel);
            if (!(dx*dx + dy*dy <= R*R)) {
                throw "Assertion failed: pixel outside the cirle!!!";
            }
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = std::abs(signedDist);
            // skip the pixels on the line
            if (dist < 0.1) continue;
            // add pixel to the corresponding half-circle
            const cv::Vec3b pixel = F.at<cv::Vec3b>(y, x);
            double b = pixel[0], g = pixel[1], r = pixel[2];
            double w = R - sqrt(dx*dx + dy*dy);
            minR = std::min(minR, r);
            minG = std::min(minG, g);
            minB = std::min(minB, b);
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
    for (int y = std::max(0, yPixel - R_LOWER); y <= std::min(F.rows, yPixel + R_LOWER); y++) {
        // solve the quadratic inequation for x: (y - yPixel)^2 + (x - xPixel)^2 <= 0
        double D = 4*(R*R - (y - yPixel)*(y - yPixel));
        int x1 = std::ceil((2*xPixel - sqrt(D)) / 2); // round up to the next integer
        int x2 = std::floor((2*xPixel + sqrt(D)) / 2); // round down to the next integer
        for (int x = std::max(0, x1); x <= std::min(F.cols, x2); x++) {
            int dy = (y - yPixel), dx = (x - xPixel);
            double signedDist = dy*unitNormY + dx*unitNormX;
            double dist = std::abs(signedDist);
            // skip the pixels on the line
            if (dist < 0.1) continue;
            // add pixel to the corresponding half-circle
            double w = R - sqrt(dx*dx + dy*dy);
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
    double ratio = std::max(area1/area2, area2/area1);
    return 1.0 - std::pow(1/ratio, 8);
}

std::tuple<cv::Mat, cv::Mat> computeBestScores(const cv::Mat &F) {
    cv::Mat S(F.size(), CV_64F), D(F.size(), CV_64F);
    // Compute the score matrix for every direction
    for (int d = 0; d < DIRECTIONS; d++) {
        // fix the normal unit vector for the direction
        double rad = d * (PI/DIRECTIONS);
        double unitY = std::sin(rad);
        double unitX = std::cos(rad);
        // compute the score matrix for the direction
        for (int y = 0; y < F.rows; y++) {
            for (int x = 0; x < F.cols; x++) {
                double score = computeScore(F, y, x, unitY, unitX);
                double oldScore = -1;
                if (d > 0) oldScore = S.at<double>(y, x);
                if (score > oldScore) {
                    S.at<double>(y, x) = score;
                    D.at<double>(y, x) = d;
                }
            }
        }
    }
    return std::make_tuple(S, D);
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
    
    // Load an RGB image
    cv::Mat F = cv::imread("../images/table.png", cv::IMREAD_COLOR);
    
    // compute the best scores for every pixel
    cv::Mat S, D;
    std::tie(S, D) = computeBestScores(F);

    // choose the candidates
    cv::Mat C = chooseCandiates(S);
    
    showImage(F);
    showMatrix(S);
    showMatrix(C);

    return 0;
}