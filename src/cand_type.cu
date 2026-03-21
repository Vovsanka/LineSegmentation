#include "cand_type.hpp"


__host__ __device__
Cand::Cand(double y, double x, int dir, double score) {
    this->y = y;
    this->x = x;
    this->dir = dir;
    this->score = score;
}

__host__ __device__
bool Cand::operator<(const Cand& otherCand) {
    return (score > otherCand.score);
}

__host__ __device__
double Cand::dist(const Cand& cand1, const Cand& cand2) {
    Vec v(cand2.y - cand1.y, cand2.x - cand1.x);
    return v.len();
}

__host__ __device__
int Cand::dirDiff(const Cand& cand1, const Cand& cand2) {
    return (cand1.dir - cand2.dir + DIRECTIONS) % DIRECTIONS;
}

__host__ __device__
double Cand::distToLine(const Cand& otherCand) const {
    Vec v(otherCand.y - y, otherCand.x - x);
    Vec u = getUnitVector(dir); // u is orthogonal to the line
    return fabs(v.dot(u));
}

__host__ __device__
thrust::tuple<uchar,uchar,uchar> Cand::getColorLab() const {
    Vec unitVector = getUnitVector(2*dir);
    int l = round(score*255.0);
    int a = round(127.5 + unitVector.x*127.5);
    int b = round(127.5 + unitVector.y*127.5);
    return thrust::make_tuple(l, a, b);
}

__host__ __device__
thrust::tuple<uchar,uchar,uchar> Cand::getColorRgb() const {
    thrust::tuple<uchar,uchar,uchar> lab = getColorLab();
    // convert LAB to RGB
    cv::Mat LAB(1, 1, CV_8UC3, cv::Vec3b(
        thrust::get<0>(lab),
        thrust::get<1>(lab),
        thrust::get<2>(lab)
    ));
    cv::Mat RGB;
    cv::cvtColor(LAB, RGB, cv::COLOR_Lab2RGB);
    cv::Vec3b pix = RGB.at<cv::Vec3b>(0, 0);
    return thrust::make_tuple(pix[0], pix[1], pix[2]);
}
