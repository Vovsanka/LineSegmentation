#include "operations.hpp"

__host__
cv::cuda::GpuMat uploadToGPU(const cv::Mat& cpuF) {
    cv::cuda::GpuMat gpuF;
    gpuF.upload(cpuF);
    return gpuF;
}

__host__
cv::Mat downloadToCPU(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    return cpuF;
}

__host__
cv::Mat convertBGRtoLab(const cv::Mat& cpuF) {
    cv::Mat labF;
    cv::cvtColor(cpuF, labF, cv::COLOR_BGR2Lab);
    return labF;
}


__host__
double computeScale(const cv::Mat& cpuF) {
    return std::min(1.0*MAX_SIDE/cpuF.cols, 1.0*MAX_SIDE/cpuF.rows);
}

__host__
cv::Mat resize(const cv::Mat& cpuF, double scale) {
    cv::Size size(std::round(scale*cpuF.cols), std::round(scale*cpuF.rows));
    cv::Mat scaledF;
    cv::resize(cpuF, scaledF, size, 0, 0, cv::INTER_CUBIC); // clamp-to-edge strategy
    return scaledF;
}

__host__
void showImage(const cv::Mat& cpuF) {
    cv::imshow("", cpuF);
    cv::waitKey(0);
}

__host__
void showImage(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    showImage(cpuF);
}

__host__
void showMatrix(const cv::Mat& cpuF) {
    cv::Mat Norm;
    cv::normalize(cpuF, Norm, 0, 255, cv::NORM_MINMAX);
    Norm.convertTo(Norm, CV_8U);
    showImage(Norm);
}

__host__
void showMatrix(const cv::cuda::GpuMat& gpuF) {
    cv::Mat cpuF;
    gpuF.download(cpuF);
    showMatrix(cpuF);
}

__host__
void showScoreDirectionMatrix(
    cv::Mat& S,
    cv::Mat& D,
    std::vector<Cand>& candidates
) {
    // 
    int demoWidth = 500;
    int demoHeight = 100;
    cv::Mat Clab(demoHeight, demoWidth, CV_8UC3);
    for (int x = 0; x < demoWidth; x++) {
        int dir = 2*round(1.0*x/demoWidth*DIRECTIONS);
        Vec unitVector = getUnitVector(dir);
        int l = 255.0;
        int a = round(127.5 + unitVector.x*127.5);
        int b = round(127.5 + unitVector.y*127.5);
        for (int y = 0; y < demoHeight; y++) {
            Clab.at<cv::Vec3b>(y, x) = cv::Vec3b(l, a, b);
        }
    }
    cv::Mat Cbgr;
    cv::cvtColor(Clab, Cbgr, cv::COLOR_Lab2BGR);
    showImage(Cbgr);
    //
    int width = S.cols;
    int height = S.rows;
    cv::Mat Mlab(height, width, CV_8UC3, cv::Scalar(0, 128, 128)); // LAB
    for (Cand& cand : candidates) {
        int y = round(cand.y);
        int x = round(cand.x);
        double score = S.at<double>(y, x);
        if (score < CAND_THRESHOLD) continue;
        int dir = D.at<int>(y, x);
        Vec unitVector = getUnitVector(2*dir);
        int l = round(score*255.0);
        int a = round(127.5 + unitVector.x*127.5);
        int b = round(127.5 + unitVector.y*127.5);
        Mlab.at<cv::Vec3b>(y, x) = cv::Vec3b(l, a, b);
    }
    cv::Mat Mbgr;
    cv::cvtColor(Mlab, Mbgr, cv::COLOR_Lab2BGR);
    showImage(Mbgr);
}

__host__
void drawClusterImage(
    int width, int height,
    const std::vector<Cand>& candidates, 
    const CandidateGraph& G,
    const std::vector<char>& edgeLabels,
    std::string name
) {
    cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    cairo_t* cr = cairo_create(surface);
    // background 
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0); 
    cairo_paint(cr); 
    // pen
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0); 
    cairo_set_line_width(cr, 0.2); 
    // draw cluster cliques using lines
    for (int k = 0; k < G.edges.size(); k++) {
        if (edgeLabels[k] > 0) continue;
        const Edge& e = G.edges[k];
        const Cand& cand1 = candidates[e.c1];
        const Cand& cand2 = candidates[e.c2];
        //
        cairo_move_to(cr, cand1.x, cand1.y); 
        cairo_line_to(cr, cand2.x, cand2.y); 
    }
    cairo_stroke(cr); 
    //
    cairo_surface_write_to_png(surface, (WORKING_STATE_DIR/(name + ".png")).string().c_str());
    //
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
}


__host__
void drawLineEdgeImage(const std::vector<Line>& lines, int width, int height, std::string name) {
    cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    cairo_t* cr = cairo_create(surface);
    // background 
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0); 
    cairo_paint(cr); 
    // pen
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0); 
    cairo_set_line_width(cr, 1.0); 
    // draw lines
    for (const Line& l : lines) {
        cairo_move_to(cr, l.x1, l.y1); 
        cairo_line_to(cr, l.x2, l.y2); 
    }
    cairo_stroke(cr); 
    //
    cairo_surface_write_to_png(surface, (WORKING_STATE_DIR/(name + ".png")).string().c_str());
    //
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
}







