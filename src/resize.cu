// #include "resize.hpp"


// cv::cuda::GpuMat resize(const cv::cuda::GpuMat& F) {
//     cv::Size size(F.cols*SCALE, F.rows*SCALE);
//     cv::cuda::GpuMat zF;
//     cv::cuda::resize(F, zF, size, 0, 0, cv::INTER_CUBIC); // clamp-to-edge strategy
//     return zF;
// }
