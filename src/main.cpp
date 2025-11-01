#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>



int main() {
    
    // 3x3 grayscale image
    cv::Mat image = (cv::Mat_<uchar>(5,5) <<
        50, 50, 50, 10, 80,
        50, 50, 10, 80, 80,
        50, 50, 10, 80, 80,
        50, 50, 10, 80, 80,
        50, 50, 10, 80, 80
    );

    cv::imshow("Harris Corners", image);
    cv::waitKey(0);

    // Compute image gradients (Sobel)
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_64F, 1, 0, 3); // dI/dx
    cv::Sobel(image, Iy, CV_64F, 0, 1, 3); // dI/dy

    // std::cout << Ix << std::endl;
    // std::cout << Iy << std::endl;
    
    // Compute structure tensor components
    cv::Mat Ix2 = Ix.mul(Ix);     // Ix^2
    cv::Mat Iy2 = Iy.mul(Iy);     // Iy^2
    cv::Mat Ixy = Ix.mul(Iy);     // Ix*Iy
    
    // 🔹 Apply Gaussian smoothing to each product
    cv::Mat Jxx, Jyy, Jxy;
    double sigma = 1.0; // standard deviation of the Gaussian

    cv::GaussianBlur(Ix2, Jxx, cv::Size(3,3), sigma);
    cv::GaussianBlur(Iy2, Jyy, cv::Size(3,3), sigma);
    cv::GaussianBlur(Ixy, Jxy, cv::Size(3,3), sigma);

    // build the structure tensor (for an image pixel)
    cv::Mat J = (cv::Mat_<double>(2, 2) << 
        Jxx.at<double>(2,2), Jxy.at<double>(2,2),
        Jxy.at<double>(2,2), Jyy.at<double>(2,2)
    );

    std::cout << "\nStructure Tensor (middle pixel):\n";
    std::cout << J << std::endl;

    // compute the eigenvalues (and eigenvectors)
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(J, eigenvalues, eigenvectors);
    double l1 = eigenvalues.at<double>(0);
    double l2 = eigenvalues.at<double>(1);
    if (l1 < l2) std::swap(l1,l2);

    std::cout << "\nEigenvalues: " << std::endl;
    std::cout << l1 << " " << l2 << std::endl;

    // classify flatness, edge, corner
    double treshold = 1000;
    if (l1 > treshold) {
        std::cout << "Likely an edge or a corner" << std::endl;
        if (l2 > treshold) {
            std::cout << "Corner rather" << std::endl;
        } else {
            std::cout << "Edge rather" << std::endl;
        }
    } else {
        std::cout << "Likely a flat region" << std::endl;
    }
    
    return 0;
}