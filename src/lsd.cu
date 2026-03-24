#include "lsd.hpp"


namespace lsd {

    std::filesystem::path imagePath;
    std::filesystem::path workingStateDir;

    void loadPreprocessImage(
        std::string originalImage_outName,
        std::string preprocessedImage_outName,
        std::string params_outName,
        bool colorImage
    ) {
        // Load an RGB image
        cv::Mat originalF = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (originalF.empty()) std::exit(1);
        
        // preprocessed original
        cv::Mat cpuF;

        // Convert the image to the LAB color space
        if (colorImage) {
            cpuF = convertBGRtoLab(originalF);
        } else {
            cpuF = convertBGRtoGrayscale(originalF);
        }

        // resize the image (to the reasonable processing size)
        // double scale = computeScale(cpuF);
        // if (scale >= 1.0) {
        //     scale = 1;
        // } else {
        //     cpuF = resizeDown(cpuF, scale);
        // }

        //
        int width = cpuF.cols;
        int height = cpuF.rows;

        std::cout << "Image: width x height = " << width << "x" << height << std::endl; 

        // save the working state
        saveMatrix(originalF, originalImage_outName);
        saveMatrix(cpuF, preprocessedImage_outName);
        saveImageParams(width, height, params_outName);
    }

    void computeThresholdCandidates(
        std::string preprocessedImage_inName,
        std::string scoreMatrix_outName,
        std::string directionMatrix_outName,
        std::string thresholdCandidates_outName,
        bool beamScore
    ) {
        // load the working state
        cv::Mat cpuF = loadMatrix(preprocessedImage_inName);

        // Upload the preprocessed matrix to GPU
        cv::cuda::GpuMat F = uploadToGPU(cpuF);

        // compute the best scores for every pixel
        cv::cuda::GpuMat S(F.size(), CV_64F);
        cv::cuda::GpuMat D(F.size(), CV_32S);
        computeBestPixelScores(F, S, D, beamScore);
        
        // download the matrices to CPU
        cv::Mat cpuS = downloadToCPU(S);
        cv::Mat cpuD = downloadToCPU(D);

        // threshold candidates
        std::vector<Cand> tCandidates = extractSortedThresholdCandidates(cpuS, cpuD);

        std::cout << "Threshold candidates: amount = " << tCandidates.size() << std::endl; 

        // save the working state
        saveMatrix(cpuS, scoreMatrix_outName);
        saveMatrix(cpuD, directionMatrix_outName);
        saveCandidates(tCandidates, thresholdCandidates_outName);
    }

    void computeIterativeCandidates(
        std::string preprocessedImage_inName,
        std::string thresholdCandidates_inName,
        std::string iterativeCandidates_outName,
        bool beamScore
    ) {
        // load the working state
        cv::Mat cpuF = loadMatrix(preprocessedImage_inName);
        std::vector<Cand> tCandidates = loadCandidates(thresholdCandidates_inName);

        // iterative search candidates
        cv::cuda::GpuMat gpuF = uploadToGPU(cpuF);
        std::vector<Cand> candidates = candidateIterativeSearch(
            cpuF.ptr<uchar>(), cpuF.step,
            gpuF,
            tCandidates,
            cpuF.cols, cpuF.rows,
            beamScore
        );

        std::cout << "Iterative candidates: amount = " << candidates.size() << std::endl; 

        // save the working state
        saveCandidates(candidates, iterativeCandidates_outName);
    }

    void buildCandidateGraph(
        std::string candidateList_inName,
        std::string candidateGraph_outName
    ) {
        // load the working state
        std::vector<Cand> candidates = loadCandidates(candidateList_inName);

        CandidateGraph G(candidates);

        std::cout << "Candidate graph: (vertices x edges) = " << G.n << "x" << G.edges.size() << std::endl;

        // save the working state
        saveCandidateGraph(G, candidateGraph_outName);
    }

    void performClustering(
        std::string candidateGraph_inName,
        std::string edgeLabels_outName
    ) {
        // load the working state
        CandidateGraph G = loadCandidateGraph(candidateGraph_inName);

        std::vector<char> edgeLabels = solveClustering(G);

        std::cout << "Clustering completed" << std::endl;

        // save the working state
        saveEdgeLabels(edgeLabels, edgeLabels_outName);
    }

    void extractLines(
        std::string candidateList_inName,
        std::string candidateGraph_inName,
        std::string edgeLabels_inName,
        std::string lines_outName
    ) {
        // load the working state
        std::vector<Cand> candidates = loadCandidates(candidateList_inName);
        CandidateGraph G = loadCandidateGraph(candidateGraph_inName);
        std::vector<char> edgeLabels = loadEdgeLabels(edgeLabels_inName);

        std::vector<Line> lines = extractLinesFromClusters(candidates, G, edgeLabels);

        std::cout << "Reconstructed lines: amount = " << lines.size() << std::endl;

        // save the working state
        saveLines(lines, lines_outName);
    }

    // void reconstructOriginalLines(
    //     std::string params_inName,
    //     std::string scaledLines_inName,
    //     std::string originalLines_outName
    // ) {
    //     // load the working state
    //     int width, height;
    //     loadImageParams(params_inName, width, height);
    //     std::vector<Line> scaledLines = loadLines(scaledLines_inName);
        
    //     if (scale > 1.0 - TOL) {
    //         // save the working state
    //         saveLines(scaledLines, originalLines_outName);
    //         return;
    //     } 

    //     double scaleY = 1.0*originalHeight/height;
    //     double scaleX = 1.0*originalWidth/width;
    //     std::vector<Line> originalLines(scaledLines.size());
    //     for (int k = 0; k < scaledLines.size(); k++) {
    //         const Line& l = scaledLines[k];
    //         originalLines[k] = Line(scaleY*l.y1, scaleX*l.x1, scaleY*l.y2, scaleX*l.x2);
    //     } 
    //     // save the working state
    //     saveLines(originalLines, originalLines_outName);
    // }

    void buildShowStateImages(
        std::string originalImage_inName,
        std::string originalImage_outName,
        std::string params_inName,
        std::string preprocessedImage_inName,
        std::string colorDirection_outName,
        std::string scoreMatrix_inName,
        std::string directionMatrix_inName,
        std::string scoreDirection_outName,
        std::string thresholdCandidates_outName,
        std::string iterativeCandidates_inName,
        std::string iterativeCandidates_outName,
        std::string candidateList_inName,
        std::string candidateGraph_inName,
        std::string candidateGraph_outName,
        std::string edgeLabels_inName,
        std::string clustering_outName,
        std::string lines_inName,
        std::string lines_outName,
        std::string originalLines_outName
    ) {
        if (!originalImage_inName.empty()) {
            cv::Mat originalF = loadMatrix(originalImage_inName);
            showImage("Original image", originalF);
            if (!originalImage_outName.empty()) {
                cv::imwrite(workingStateDir/(originalImage_outName + ".png"), originalF);
            }
        }
        if (!originalImage_inName.empty()) {
            int width, height;
            loadImageParams(params_inName, width, height);
            std::cout << "Image size: " << width << "x" << height << std::endl;    
        }
        if (!preprocessedImage_inName.empty()) {
            cv::Mat preprocessedF = loadMatrix(preprocessedImage_inName);
            showImage("Preprocessed image", preprocessedF);
        }
        if (!colorDirection_outName.empty()) {
            cv::Mat P = buildColorDirectionMap();
            cv::imwrite(workingStateDir/(colorDirection_outName + ".png"), P);
            showMatrix("Color-direction map", P);
        }
        if (!scoreMatrix_inName.empty() && !directionMatrix_inName.empty()) {
            cv::Mat S = loadMatrix(scoreMatrix_inName);
            cv::Mat D = loadMatrix(directionMatrix_inName);
            if (!scoreDirection_outName.empty()) {
                cv::Mat R = buildScoreDirectionMatrix(S, D);
                cv::imwrite(workingStateDir/(scoreDirection_outName + ".png"), R);
                showMatrix("Score-direction matrix", R);
            }
            if (!thresholdCandidates_outName.empty()) {
                cv::Mat T = buildScoreDirectionMatrix(S, D, CAND_THRESHOLD);
                cv::imwrite(workingStateDir/(thresholdCandidates_outName + ".png"), T);
                showMatrix("Theshold candidates", T);
            }
        }
        if (!params_inName.empty() && !iterativeCandidates_inName.empty() && !iterativeCandidates_outName.empty()) {
            int width, height;
            loadImageParams(params_inName, width, height);
            std::vector<Cand> candidates = loadCandidates(iterativeCandidates_inName);
            buildGraphImage(iterativeCandidates_outName, width, height, candidates);
            cv::Mat F = cv::imread(workingStateDir/(iterativeCandidates_outName + ".png"), cv::IMREAD_COLOR);
            showImage("Iterative candidates", F);
        }
        if (!params_inName.empty() && !candidateList_inName.empty() && !candidateGraph_inName.empty()) {
            int width, height;
            loadImageParams(params_inName, width, height);
            std::vector<Cand> candidates = loadCandidates(candidateList_inName);
            CandidateGraph cgraph = loadCandidateGraph(candidateGraph_inName);
            if (!candidateGraph_outName.empty()) {
                buildGraphImage(candidateGraph_outName, width, height, candidates, cgraph);
                cv::Mat cgraphF = cv::imread(workingStateDir/(candidateGraph_outName + ".png"), cv::IMREAD_COLOR);
                showImage("Candidate graph", cgraphF);
            }
            if (!edgeLabels_inName.empty() && !clustering_outName.empty()) {
                std::vector<char> edgeLabels = loadEdgeLabels(edgeLabels_inName);
                buildGraphImage(clustering_outName, width, height, candidates, cgraph, edgeLabels);
                cv::Mat clusteringF = cv::imread(workingStateDir/(clustering_outName + ".png"), cv::IMREAD_COLOR);
                showImage("Clustering", clusteringF);
            }
        }
        if (!params_inName.empty() && !lines_inName.empty() && !lines_outName.empty()) {
            int width, height;
            loadImageParams(params_inName, width, height);
            std::vector<Line> lines = loadLines(lines_inName);
            buildLineImage(lines_outName, width, height, lines);
            cv::Mat linesF = cv::imread(workingStateDir/(lines_outName + ".png"), cv::IMREAD_COLOR);
            showImage("Reconstructed lines", linesF);
            if (!originalImage_outName.empty()) {
                buildLineImage(originalLines_outName, width, height, lines, originalImage_outName);
                cv::Mat originalLinesF = cv::imread(workingStateDir/(originalLines_outName + ".png"), cv::IMREAD_COLOR);
                showImage("Original with lines", originalLinesF);
            }
        }
    }

    //////////

    cv::Mat buildScoreDirectionMatrix(
        cv::Mat& S,
        cv::Mat& D,
        double threshold
    ) {
        //
        int width = S.cols;
        int height = S.rows;
        cv::Mat Mlab(height, width, CV_8UC3, cv::Scalar(0, 128, 128)); // LAB
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double score = S.at<double>(y, x);
                if (score < threshold) continue;
                int dir = D.at<int>(y, x);
                thrust::tuple<uchar,uchar,uchar> lab = getDirColorLab(dir);
                int l = thrust::get<0>(lab);
                int a = thrust::get<1>(lab);
                int b = thrust::get<2>(lab);
                Mlab.at<cv::Vec3b>(y, x) = cv::Vec3b(l, a, b);
            }
        }
        cv::Mat Mbgr;
        cv::cvtColor(Mlab, Mbgr, cv::COLOR_Lab2BGR);
        return Mbgr;
    }

    cv::Mat buildColorDirectionMap() {
        // /// demo for direction-color circular mapping
        // int demoWidth = 500;
        // int demoHeight = 100;
        // cv::Mat Clab(demoHeight, demoWidth, CV_8UC3);
        // for (int x = 0; x < demoWidth; x++) {
        //     int dir = round(1.0*x/demoWidth*DIRECTIONS);
        //     thrust::tuple<uchar,uchar,uchar> lab = getDirColorLab(dir);
        //     int l = thrust::get<0>(lab);
        //     int a = thrust::get<1>(lab);
        //     int b = thrust::get<2>(lab);
        //     for (int y = 0; y < demoHeight; y++) {
        //         Clab.at<cv::Vec3b>(y, x) = cv::Vec3b(l, a, b);
        //     }
        // }
        // cv::Mat Cbgr;
        // cv::cvtColor(Clab, Cbgr, cv::COLOR_Lab2BGR);
        // return Cbgr;
        /// demo for direction-color circular mapping
        int demoWidth = 500;
        int demoHeight = 100;
        cv::Mat Cbgr(demoHeight, demoWidth, CV_8UC3);
        for (int x = 0; x < demoWidth; x++) {
            int dir = round(1.0*x/demoWidth*DIRECTIONS);
            thrust::tuple<uchar,uchar,uchar> rgb = getDirColorRgb(dir);
            int r = thrust::get<0>(rgb);
            int g = thrust::get<1>(rgb);
            int b = thrust::get<2>(rgb);
            for (int y = 0; y < demoHeight; y++) {
                Cbgr.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
            }
        }
        return Cbgr;
    }

    void buildGraphImage(
        std::string& name, 
        int width, int height,
        const std::vector<Cand>& candidates, 
        const CandidateGraph& cgraph,
        const std::vector<char>& edgeLabels
    ) { 
        //
        std::vector<thrust::tuple<double,double,double>> colorMapping(candidates.size());
        if (!edgeLabels.empty()) {
            std::mt19937 rng(12345); // deterministic seed
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            //
            std::vector<std::vector<int>> clusters = retrieveClusters(cgraph, edgeLabels);
            for (const std::vector<int>& cluster : clusters) {
                thrust::tuple<double,double,double> color = thrust::make_tuple(dist(rng), dist(rng), dist(rng));
                for (int node : cluster) {
                    colorMapping[node] = color;
                }
            }
        }
        //
        cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
        cairo_t* cr = cairo_create(surface);
        // background 
        cairo_set_source_rgb(cr, 0.0, 0.0, 0.0); 
        cairo_paint(cr); 
        // draw candidate points
        cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);                    
        for (int k = 0; k < candidates.size(); k++) {
            const Cand& cand = candidates[k];
            if (cgraph.edges.empty()) {
                thrust::tuple<uchar,uchar,uchar> rgb = getDirColorRgb(cand.dir);
                double r = thrust::get<0>(rgb)/255.0;
                double g = thrust::get<1>(rgb)/255.0;
                double b = thrust::get<2>(rgb)/255.0;
                cairo_set_source_rgb(cr, r, g, b); 
            } else if (!edgeLabels.empty()) {
                double r = thrust::get<0>(colorMapping[k]);
                double g = thrust::get<1>(colorMapping[k]);
                double b = thrust::get<2>(colorMapping[k]);
                cairo_set_source_rgb(cr, r, g, b); 
            }
            cairo_arc(cr, cand.x, cand.y, 1.0, 1, 2*PI);
            cairo_fill(cr);
        }
        // draw all or labeled edges
        cairo_set_line_width(cr, 0.2); 
        for (int k = 0; k < cgraph.edges.size(); k++) {
            const Edge& e = cgraph.edges[k];
            //
            if (edgeLabels.empty()) {
                if (abs(e.w) < TOL) continue;
                if (e.w > 0) {
                    cairo_set_source_rgb(cr, 0, 0, 1.0); 
                    cairo_set_line_width(cr, 1); 
                } else {
                    cairo_set_source_rgb(cr, 1.0, 0, 0); 
                    cairo_set_line_width(cr, 0.2); 
                }
            } else {
                break;
            }
            //
            const Cand& cand1 = candidates[e.c1];
            const Cand& cand2 = candidates[e.c2];
            //
            cairo_move_to(cr, cand1.x, cand1.y); 
            cairo_line_to(cr, cand2.x, cand2.y); 
            cairo_stroke(cr); 
        }
        //
        cairo_surface_write_to_png(surface, (workingStateDir/(name + ".png")).string().c_str());
        //
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
    }

    void buildLineImage(
        std::string& name, 
        int width, int height,
        const std::vector<Line>& lines,
        std::string originalName
    ) {
        //
        cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
        cairo_t* cr = cairo_create(surface);
        //
        std::vector<thrust::tuple<double,double,double>> colorMapping(lines.size());
        if (!originalName.empty()) {
            cairo_surface_t* bg = cairo_image_surface_create_from_png((workingStateDir/(originalName + ".png")).string().c_str());
            cairo_set_source_surface(cr, bg, 0, 0);
            cairo_paint(cr); 
        } else {
            cairo_set_source_rgb(cr, 0.0, 0.0, 0.0); 
            cairo_paint(cr); 
            //
            std::mt19937 rng(12345); // deterministic seed
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            //
            for (int k = 0; k < lines.size(); k++) {
                thrust::tuple<double,double,double> color = thrust::make_tuple(dist(rng), dist(rng), dist(rng));
                colorMapping[k] = color;
            }
        }
        // draw lines
        cairo_set_line_width(cr, 0.2); 
       for (int k = 0; k < lines.size(); k++) {
            const Line& line = lines[k];
            //
            if (!originalName.empty()) {
                cairo_set_source_rgb(cr, 0, 1.0, 0); 
                cairo_set_line_width(cr, 3); 
            } else {
                double r = thrust::get<0>(colorMapping[k]);
                double g = thrust::get<1>(colorMapping[k]);
                double b = thrust::get<2>(colorMapping[k]);
                cairo_set_source_rgb(cr, r, g, b); 
                cairo_set_line_width(cr, 1); 
            }
            //
            cairo_move_to(cr, line.x1, line.y1); 
            cairo_line_to(cr, line.x2, line.y2); 
            cairo_stroke(cr); 
        }
        //
        cairo_surface_write_to_png(surface, (workingStateDir/(name + ".png")).string().c_str());
        //
        cairo_destroy(cr);
        cairo_surface_destroy(surface);
    }   

    ////////////////////

    void saveMatrix(const cv::Mat& M, std::string name) {
        std::ofstream out(workingStateDir/(name + ".bin"), std::ios::binary);
        int rows = M.rows, cols = M.cols, type = M.type();
        out.write((char*)&rows, sizeof(int)); 
        out.write((char*)&cols, sizeof(int)); 
        out.write((char*)&type, sizeof(int)); 
        out.write((char*)M.data, M.total()*M.elemSize());
    }

    cv::Mat loadMatrix(std::string name) {
        std::ifstream in(workingStateDir/(name + ".bin"), std::ios::binary); 
        int rows, cols, type; 
        in.read((char*)&rows, sizeof(int)); 
        in.read((char*)&cols, sizeof(int)); 
        in.read((char*)&type, sizeof(int)); 
        //
        cv::Mat M(rows, cols, type); 
        in.read((char*)M.data, M.total()*M.elemSize());
        return M;
    }

    void saveImageParams(
        int width, int height,
        std::string& name
    ) {
        std::ofstream out(workingStateDir / (name + ".txt"));
        out << width << " " << height << "\n";
    }

    void loadImageParams(
        std::string& name,
        int& width, int& height
    ) {
        std::ifstream in(workingStateDir / (name + ".txt"));
        in >> width >> height;
    }

    void saveCandidates(const std::vector<Cand>& candidates, std::string& name) {
        std::ofstream out(workingStateDir / (name + ".txt"));

        out << candidates.size() << "\n";

        for (const Cand& c : candidates) {
            out << c.y << " "
                << c.x << " "
                << c.dir << " "
                << c.score << "\n";
        }
    }

    std::vector<Cand> loadCandidates(std::string& name) {
        std::ifstream in(workingStateDir / (name + ".txt"));

        std::size_t count = 0;
        in >> count;

        std::vector<Cand> candidates(count);

        for (std::size_t i = 0; i < count; i++) {
            in >> candidates[i].y
            >> candidates[i].x
            >> candidates[i].dir
            >> candidates[i].score;
        }

        return candidates;
    }

    void saveCandidateGraph(const CandidateGraph& G, std::string& name) {
        std::ofstream out(workingStateDir / (name + ".txt"));

        out << G.n << "\n";
        out << G.edges.size() << "\n";

        for (const Edge& e : G.edges) {
            out << e.c1 << " "
                << e.c2 << " "
                << e.w  << "\n";
        }
    }

    CandidateGraph loadCandidateGraph(std::string& name) {
        std::ifstream in(workingStateDir / (name + ".txt"));

        CandidateGraph G;

        std::size_t vertexCount = 0;
        std::size_t edgeCount = 0;

        in >> vertexCount;
        in >> edgeCount;

        G.n = vertexCount;
        G.edges.resize(edgeCount);

        for (std::size_t i = 0; i < edgeCount; i++) {
            in >> G.edges[i].c1
            >> G.edges[i].c2
            >> G.edges[i].w;
        }

        return G;
    }

    void saveEdgeLabels(const std::vector<char>& edgeLabels, std::string& name) {
        std::ofstream out(workingStateDir / (name + ".txt"));

        out << edgeLabels.size() << "\n";

        for (char c : edgeLabels) {
            out << int(c) << "\n";   // write 0 or 1 as integer
        }
    }

    std::vector<char> loadEdgeLabels(std::string& name) {
        std::ifstream in(workingStateDir / (name + ".txt"));

        std::size_t count = 0;
        in >> count;

        std::vector<char> edgeLabels(count);

        for (std::size_t i = 0; i < count; i++) {
            int temp;
            in >> temp;              // read 0 or 1
            edgeLabels[i] = char(temp);
        }

        return edgeLabels;
    }

    void saveLines(const std::vector<Line>& lines, std::string& name) {
        std::ofstream out(workingStateDir / (name + ".txt"));

        out << lines.size() << "\n";

        for (const Line& L : lines) {
            out << L.y1 << " "
                << L.x1 << " "
                << L.y2 << " "
                << L.x2 << "\n";
        }
    }


    std::vector<Line> loadLines(std::string& name) {
        std::ifstream in(workingStateDir / (name + ".txt"));

        std::size_t count = 0;
        in >> count;

        std::vector<Line> lines(count);

        for (std::size_t i = 0; i < count; i++) {
            in >> lines[i].y1
            >> lines[i].x1
            >> lines[i].y2
            >> lines[i].x2;
        }

        return lines;
    }


    

}
