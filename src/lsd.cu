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
        // showImage(originalF);
        
        // preprocessed original
        cv::Mat cpuF;

        // Convert the image to the LAB color space
        if (colorImage) {
            cpuF = convertBGRtoLab(originalF);
        } else {
            cpuF = convertBGRtoGrayscale(originalF);
        }

        // resize the image (to the reasonable processing size)
        double scale = computeScale(cpuF);
        if (scale >= 1.0) {
            scale = 1;
        } else {
            cpuF = resizeDown(cpuF, scale);
        }
        // showImage(cpuF);
        
        // print image size
        std::cout << "Original image size: " << originalF.cols << "x" << originalF.rows << std::endl; 
        std::cout << "Scale factor: " << scale << std::endl;
        std::cout << "cpu image size: " << cpuF.cols << "x" << cpuF.rows << std::endl;    

        // save the working state
        saveMatrix(originalF, originalImage_outName);
        saveMatrix(cpuF, preprocessedImage_outName);
        saveImageParams(originalF.cols, originalF.rows, scale, cpuF.cols, cpuF.rows, params_outName);
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

        //// debug start
        computeGrayScore(cpuF.ptr<uchar>(), cpuF.step, 50, 320, 0, cpuF.cols, cpuF.rows);
        //// debug end

        // Upload the preprocessed matrix to GPU
        cv::cuda::GpuMat F = uploadToGPU(cpuF);

        // compute the best scores for every pixel
        cv::cuda::GpuMat S(F.size(), CV_64F);
        cv::cuda::GpuMat D(F.size(), CV_32S);
        computeBestPixelScores(F, S, D, beamScore);
        // showMatrix(S);
        
        // download the matrices to CPU
        cv::Mat cpuS = downloadToCPU(S);
        cv::Mat cpuD = downloadToCPU(D);

        // threshold candidates
        std::vector<Cand> tCandidates = extractSortedThresholdCandidates(cpuS, cpuD);

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

        // save the working state
        saveLines(lines, lines_outName);
    }

    void reconstructOriginalLines(
        std::string params_inName,
        std::string scaledLines_inName,
        std::string originalLines_outName
    ) {
        // load the working state
        int originalWidth, originalHeight;
        double scale;
        int width, height;
        loadImageParams(params_inName, originalWidth, originalHeight, scale, width, height);
        std::vector<Line> scaledLines = loadLines(scaledLines_inName);
        
        if (scale > 1.0 - TOL) {
            // save the working state
            saveLines(scaledLines, originalLines_outName);
            return;
        } 

        double scaleY = 1.0*originalHeight/height;
        double scaleX = 1.0*originalWidth/width;
        std::vector<Line> originalLines(scaledLines.size());
        for (int k = 0; k < scaledLines.size(); k++) {
            const Line& l = scaledLines[k];
            originalLines[k] = Line(scaleY*l.y1, scaleX*l.x1, scaleY*l.y2, scaleX*l.x2);
        } 
        // save the working state
        saveLines(originalLines, originalLines_outName);
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
        int originalWidth, int originalHeight,
        double scale,
        int width, int height,
        std::string& name
    ) {
        std::ofstream out(workingStateDir / (name + ".txt"));

        out << originalWidth  << "\n"
            << originalHeight << "\n"
            << scale          << '\n'
            << width          << "\n"
            << height         << "\n";
    }

    void loadImageParams(
        std::string& name,
        int& originalWidth, int& originalHeight,
        double &scale,
        int& width, int& height
    ) {
        std::ifstream in(workingStateDir / (name + ".txt"));

        in >> originalWidth
           >> originalHeight
           >> scale
           >> width
           >> height;
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
