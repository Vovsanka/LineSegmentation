#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>


int Image[3][3] = {
    {50, 10, 80},
    {50, 10, 80},
    {50, 10, 80}
};


int main() {
    
    // 3x3 grayscale image
    cv::Mat image = (cv::Mat_<uchar>(3,3) << 50, 10, 80,
                                              50, 10, 80,
                                              50, 10, 80);

    // Compute image gradients (Sobel)
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_64F, 1, 0, 3); // dI/dx
    cv::Sobel(image, Iy, CV_64F, 0, 1, 3); // dI/dy

    
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

    std::cout << "Structure Tensor (middle):\n";
    std::cout << "[ " << Jxx.at<double>(1,1) << " , " << Jxy.at<double>(1,1) << " ]\n";
    std::cout << "[ " << Jxy.at<double>(1,1) << " , " << Jyy.at<double>(1,1) << " ]\n";

    return 0;
}