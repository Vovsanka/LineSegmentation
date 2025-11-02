#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>


void showMat(const cv::Mat &M) {
    cv::Mat mNorm;
    cv::normalize(M, mNorm, 0, 255, cv::NORM_MINMAX);
    mNorm.convertTo(mNorm, CV_8U);
    cv::imshow("Matrix", mNorm);
    cv::waitKey(0);
}




int main() {
    
    // Configuration start
    const int SOBEL_KSIZE = 3; // kernel size of the sobel filter
    double SIGMA = 1.0; // standard deviation of the Gaussian
    cv::Size NEIGHBORHOOD = cv::Size(3, 3); // neighborhood for the Gausssian smoothing
    double THRESHOLD_EIGENVALUE = 0.25; // *mean_eigenvalue1 (determine large/small)
    // Configuration end

    // Load a grayscale image
    cv::Mat image = cv::imread("../images/table.png", cv::IMREAD_GRAYSCALE);

    // Compute image gradients (Sobel filter)
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_64F, 1, 0, SOBEL_KSIZE); // dI/dx
    cv::Sobel(image, Iy, CV_64F, 0, 1, SOBEL_KSIZE); // dI/dy
    
    // Compute structure tensor components
    cv::Mat Ix2 = Ix.mul(Ix);     // Ix^2
    cv::Mat Iy2 = Iy.mul(Iy);     // Iy^2
    cv::Mat Ixy = Ix.mul(Iy);     // Ix*Iy
    
    // Apply Gaussian smoothing to each product
    cv::Mat Jxx, Jyy, Jxy;

    cv::GaussianBlur(Ix2, Jxx, NEIGHBORHOOD, SIGMA);
    cv::GaussianBlur(Iy2, Jyy, NEIGHBORHOOD, SIGMA);
    cv::GaussianBlur(Ixy, Jxy, NEIGHBORHOOD, SIGMA);

    // build the structure tensors for each pixel and compute the eigenvalues
    cv::Mat L1(image.size(), CV_64F);
    cv::Mat L2(image.size(), CV_64F);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
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

    // classify edges and corners
    cv::Scalar meanL1 = cv::mean(L1);
    double threshold = THRESHOLD_EIGENVALUE * meanL1[0];
    cv::Mat E(image.size(), CV_64F);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            double l1 = L1.at<double>(y, x);
            double l2 = L2.at<double>(y, x);
            if (l1 > threshold) { // edge or corner
                if (l2 > threshold) { // corner
                    E.at<double>(y,x) = 2;
                } else {
                    E.at<double>(y,x) = 1;
                }
                // E.at<double>(y,x) = 1;
            } else {
                E.at<double>(y,x) = 0;
            }
        }
    }

    // Direction projection and score example
    double yNorm = 1.0, xNorm = 0; // normal unit vector for a direction
    cv::Mat S(image.size(), CV_64F);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            double jxx = Jxx.at<double>(y, x);
            double jyy = Jyy.at<double>(y, x);
            double jxy = Jxy.at<double>(y, x);
            double l1 = L1.at<double>(y, x);
            double l2 = L2.at<double>(y, x);
            double e = E.at<double>(y, x);
            double score = 0;
            if (e > 0) {
                double pr = yNorm*yNorm*jxx + 2*yNorm*xNorm*jxy + xNorm*xNorm*jyy;
                score = pr / l1;
            }
            S.at<double>(y, x) = score;
        }
    }
    showMat(S);
    
    return 0;
}