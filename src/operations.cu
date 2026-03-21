#include "operations.hpp"

cv::cuda::GpuMat uploadToGPU(const cv::Mat& cpuF) {
    cv::cuda::GpuMat gpuF;
    gpuF.upload(cpuF);
    return gpuF;
}

cv::Mat downloadToCPU(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    return cpuF;
}

cv::Mat convertBGRtoLab(const cv::Mat& cpuF) {
    cv::Mat labF;
    cv::cvtColor(cpuF, labF, cv::COLOR_BGR2Lab);
    return labF;
}

cv::Mat convertBGRtoGrayscale(const cv::Mat& cpuF) {
    cv::Mat grayF;
    cv::cvtColor(cpuF, grayF, cv::COLOR_BGR2GRAY);
    return grayF;
}


double computeScale(const cv::Mat& cpuF) {
    return std::min(1.0*MAX_SIDE/cpuF.cols, 1.0*MAX_SIDE/cpuF.rows);
}

cv::Mat resizeDown(const cv::Mat& cpuF, double scale) {
    cv::Size size(std::round(scale*cpuF.cols), std::round(scale*cpuF.rows));
    cv::Mat scaledF;
    cv::resize(cpuF, scaledF, size, 0, 0, cv::INTER_AREA); // clamp-to-edge strategy
    return scaledF;
}

void showImage(const cv::Mat& cpuF) {
    cv::imshow("", cpuF);
    cv::waitKey(0);
}

void showImage(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    showImage(cpuF);
}

void showMatrix(const cv::Mat& cpuF) {
    cv::Mat Norm;
    cv::normalize(cpuF, Norm, 0, 255, cv::NORM_MINMAX);
    Norm.convertTo(Norm, CV_8U);
    showImage(Norm);
}

void showMatrix(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    showMatrix(cpuF);
}


// void drawClusterImage(
//     int width, int height,
//     const std::vector<Cand>& candidates, 
//     const CandidateGraph& G,
//     const std::vector<char>& edgeLabels,
//     std::string name
// ) {
//     cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
//     cairo_t* cr = cairo_create(surface);
//     // background 
//     cairo_set_source_rgb(cr, 0.0, 0.0, 0.0); 
//     cairo_paint(cr); 
//     // pen
//     cairo_set_source_rgb(cr, 1.0, 1.0, 1.0); 
//     cairo_set_line_width(cr, 0.2); 
//     // draw cluster cliques using lines
//     for (int k = 0; k < G.edges.size(); k++) {
//         if (edgeLabels[k] > 0) continue;
//         const Edge& e = G.edges[k];
//         const Cand& cand1 = candidates[e.c1];
//         const Cand& cand2 = candidates[e.c2];
//         //
//         cairo_move_to(cr, cand1.x, cand1.y); 
//         cairo_line_to(cr, cand2.x, cand2.y); 
//     }
//     cairo_stroke(cr); 
//     //
//     cairo_surface_write_to_png(surface, (workingStateDir/(name + ".png")).string().c_str());
//     //
//     cairo_destroy(cr);
//     cairo_surface_destroy(surface);
// }


// void drawLineEdgeImage(const std::vector<Line>& lines, int width, int height, std::string name) {
//     cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
//     cairo_t* cr = cairo_create(surface);
//     // background 
//     cairo_set_source_rgb(cr, 0.0, 0.0, 0.0); 
//     cairo_paint(cr); 
//     // pen
//     cairo_set_source_rgb(cr, 1.0, 1.0, 1.0); 
//     cairo_set_line_width(cr, 1.0); 
//     // draw lines
//     for (const Line& l : lines) {
//         cairo_move_to(cr, l.x1, l.y1); 
//         cairo_line_to(cr, l.x2, l.y2); 
//     }
//     cairo_stroke(cr); 
//     //
//     cairo_surface_write_to_png(surface, (workingStateDir/(name + ".png")).string().c_str());
//     //
//     cairo_destroy(cr);
//     cairo_surface_destroy(surface);
// }







