#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>


int Image[3][3] = {
    {50, 10, 80},
    {50, 10, 80},
    {50, 10, 80}
};


int main() {
    std::cout << Image[1][1] << std::endl;
    return 0;
}