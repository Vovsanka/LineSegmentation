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
        saveImageParams(originalF.cols, originalF.rows, cpuF.cols, cpuF.rows, params_outName);
    }

    void computeThresholdCandidates(
        std::string preprocessedImage_inName,
        std::string scoreMatrix_outName,
        std::string directionMatrix_outName,
        std::string candidateList_outName,
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
        saveCandidates(tCandidates, candidateList_outName);
    }

    void computeIterativeCandidates(
        std::string preprocessedImage_inName,
        std::string candidateList_inName,
        std::string candidateList_outName,
        bool beamScore
    ) {
        // load the working state
        cv::Mat cpuF = loadMatrix(preprocessedImage_inName);
        std::vector<Cand> tCandidates = loadCandidates(candidateList_inName);

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
        saveCandidates(candidates, candidateList_outName);
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
        std::string lines_inName,
        std::string lines_outName
    ) {
        // load the working state
        int originalWidth, originalHeight;
        int width, height;
        std::tie(originalWidth, originalHeight, width, height) = loadImageParams(params_inName);
        std::vector<Line> lines = loadLines(lines_inName);
        
        double scaleY = 1.0*originalHeight/height;
        double scaleX = 1.0*originalWidth/width;
        std::vector<Line> originalLines(lines.size());
        for (int k = 0; k < lines.size(); k++) {
            const Line& l = lines[k];
            originalLines[k] = Line(scaleY*l.y1, scaleX*l.x1, scaleY*l.y2, scaleX*l.x2);
        } 

        // save the working state
        saveLines(originalLines, lines_outName);
    }



    

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
        int width, int height,
        std::string name
    ) {
        std::ofstream out(workingStateDir/(name + ".bin"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(&originalWidth), sizeof(originalWidth));
        out.write(reinterpret_cast<const char*>(&originalHeight), sizeof(originalHeight));
        out.write(reinterpret_cast<const char*>(&width), sizeof(width));
        out.write(reinterpret_cast<const char*>(&height), sizeof(height));
    }

    std::tuple<int,int,int,int> loadImageParams(std::string name) {
        std::ifstream in(workingStateDir/(name + ".bin"), std::ios::binary);
        int originalWidth = 0, originalHeight = 0;
        int width = 0, height = 0;
        in.read(reinterpret_cast<char*>(&originalWidth), sizeof(originalWidth));
        in.read(reinterpret_cast<char*>(&originalHeight), sizeof(originalHeight));
        in.read(reinterpret_cast<char*>(&width), sizeof(width));
        in.read(reinterpret_cast<char*>(&height), sizeof(height));
        return std::make_tuple(originalWidth, originalHeight, width, height);
    }

    void saveCandidates(const std::vector<Cand>& candidates, std::string name) {
        std::ofstream out(workingStateDir/(name + ".bin"), std::ios::binary);
        std::size_t count = candidates.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        if (count > 0) {
            out.write(reinterpret_cast<const char*>(candidates.data()), count*sizeof(Cand)); 
        }
    }

    std::vector<Cand> loadCandidates(std::string name) {
        std::ifstream in(workingStateDir/(name + ".bin"), std::ios::binary);
        std::size_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        std::vector<Cand> candidates(count);
        if (count > 0) { 
            in.read(reinterpret_cast<char*>(candidates.data()), count*sizeof(Cand)); 
        } 
        return candidates;
    }

    void saveCandidateGraph(const CandidateGraph& G, std::string name) {
        std::ofstream out(workingStateDir/(name + ".bin"), std::ios::binary);
        std::size_t vertexCount = G.n;
        std::size_t edgeCount = G.edges.size();
        out.write(reinterpret_cast<const char*>(&vertexCount), sizeof(vertexCount));
        out.write(reinterpret_cast<const char*>(&edgeCount), sizeof(edgeCount));
        if (edgeCount > 0) {
            out.write(reinterpret_cast<const char*>(G.edges.data()), edgeCount*sizeof(Edge)); 
        }
    }

    CandidateGraph loadCandidateGraph(std::string name) {
        std::ifstream in(workingStateDir/(name + ".bin"), std::ios::binary);
        std::size_t vertexCount = 0;
        std::size_t edgeCount = 0;
        in.read(reinterpret_cast<char*>(&vertexCount), sizeof(vertexCount));
        in.read(reinterpret_cast<char*>(&edgeCount), sizeof(edgeCount));
        std::vector<Edge> edges(edgeCount);
        if (edgeCount > 0) { 
            in.read(reinterpret_cast<char*>(edges.data()), edgeCount*sizeof(Edge)); 
        } 
        //
        CandidateGraph G;
        G.n = vertexCount;
        G.edges = edges;
        return G;
    }

    void saveEdgeLabels(const std::vector<char>& edgeLabels, std::string name) {
        std::ofstream out(workingStateDir/(name + ".bin"), std::ios::binary);
        std::size_t edgeCount = edgeLabels.size();
        out.write(reinterpret_cast<const char*>(&edgeCount), sizeof(edgeCount));
        if (edgeCount > 0) {
            out.write(reinterpret_cast<const char*>(edgeLabels.data()), edgeCount); 
        }
    }

    std::vector<char> loadEdgeLabels(std::string name) {
        std::ifstream in(workingStateDir/(name + ".bin"), std::ios::binary);
        std::size_t edgeCount = 0;
        in.read(reinterpret_cast<char*>(&edgeCount), sizeof(edgeCount));
        std::vector<char> edgeLabels(edgeCount);
        if (edgeCount > 0) { 
            in.read(reinterpret_cast<char*>(edgeLabels.data()), edgeCount); 
        } 
        return edgeLabels;
    }

    void saveLines(const std::vector<Line>& lines, std::string name) {
        std::ofstream out(workingStateDir/(name + ".bin"), std::ios::binary);
        std::size_t lineCount = lines.size();
        out.write(reinterpret_cast<const char*>(&lineCount), sizeof(lineCount));
        if (lineCount > 0) {
            out.write(reinterpret_cast<const char*>(lines.data()), lineCount*sizeof(Line)); 
        }
    }

    std::vector<Line> loadLines(std::string name) {
        std::ifstream in(workingStateDir/(name + ".bin"), std::ios::binary);
        std::size_t lineCount = 0;
        in.read(reinterpret_cast<char*>(&lineCount), sizeof(lineCount));
        std::vector<Line> lines(lineCount);
        if (lineCount > 0) { 
            in.read(reinterpret_cast<char*>(lines.data()), lineCount*sizeof(Line)); 
        } 
        return lines;
    }

}
