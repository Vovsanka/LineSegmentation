#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>



int main() {
    
    // Configuration start
    const int SOBEL_KSIZE = 3; // kernel size of the sobel filter
    double SIGMA = 1.0; // standard deviation of the Gaussian
    cv::Size NEIGHBORHOOD = cv::Size(3, 3); // neighborhood for the Gausssian smoothing
    double THRESHOLD_EIGENVALUE = 0.01; // *max_eigenvalue (determine large/small)
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
    
    // 🔹 Apply Gaussian smoothing to each product
    cv::Mat Jxx, Jyy, Jxy;

    cv::GaussianBlur(Ix2, Jxx, NEIGHBORHOOD, SIGMA);
    cv::GaussianBlur(Iy2, Jyy, NEIGHBORHOOD, SIGMA);
    cv::GaussianBlur(Ixy, Jxy, NEIGHBORHOOD, SIGMA);

    // build the structure tensors for each pixel and compute the eigenvalues
    cv::Mat L1(image.size(), CV_64F);
    cv::Mat L2(image.size(), CV_64F);
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

    // classify edges and corners
    double maxL1, minL1;
    cv::minMaxLoc(L1, &minL1, &maxL1);
    double threshold = THRESHOLD_EIGENVALUE * maxL1;
    cv::Mat E(image.size(), CV_64F);
    for (int y = 0; y < Jxx.rows; y++) {
        for (int x = 0; x < Jxx.cols; x++) {
            double l1 = L1.at<double>(y, x);
            double l2 = L2.at<double>(y, x);
            if (l1 > threshold) { // edge or corner
                // if (l2 > threshold) { // corner
                //     E.at<double>(y,x) = 2;
                // } else {
                //     E.at<double>(y,x) = 1;
                // }
                E.at<double>(y,x) = 1;
            } else {
                E.at<double>(y,x) = 0;
            }
        }
    }


    cv::Mat eNorm;
    cv::normalize(E, eNorm, 0, 255, cv::NORM_MINMAX);
    eNorm.convertTo(eNorm, CV_8U);

    cv::imshow("Table", image);
    cv::imshow("Lambda1", eNorm);
    cv::waitKey(0);
    
    return 0;
}