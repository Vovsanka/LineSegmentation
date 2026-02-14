#include "iterative.hpp"


__host__ __device__
Cand upgradeCandidate(
    const uchar* F, size_t Fstep,
    cv::cuda::GpuMat& gpuF,
    Cand cand,
    int width, int height
) {
    //
    Vec unitNorm = getUnitVector(cand.dir);
    //
    double bestScore = cand.score;
    double bestY = cand.y, bestX = cand.x;
    for (int k = -UP_COUNT; k <= UP_COUNT; k++) {
        double y = cand.y + k*UP_STEP*unitNorm.y;
        double x = cand.x + k*UP_STEP*unitNorm.x;
        //
        double score = computeLabScore(F, Fstep, y, x, cand.dir, width, height);
        //
        if (score > bestScore) {
            bestScore = score;
            bestY = y;
            bestX = x;
        }
    }
    //
    if (abs(bestY - cand.y) < UP_STEP && abs(bestX - cand.x) < UP_STEP) return cand;
    // arrive at this point iff the score got upgraded (max score limited => no endless loop)
    Cand bestScoreDir = computeBestPixelCandidate(gpuF, bestY, bestX);
    double bestDir = bestScoreDir.dir;
    return upgradeCandidate(F, Fstep, gpuF, Cand(bestY, bestX, bestDir, bestScore), width, height);
}


__host__
bool isBlocked(const uchar* B, size_t Bstep, double y, double x, int width, int height) {
    int closestY = std::round(y);
    int closestX = std::round(x);
    if (closestY < 0 || height <= closestY) return true;
    if (closestX < 0 || width <= closestX) return true;
    return (cell<uchar>(B, Bstep, closestY, closestX) != 0);
}

__host__
void setBlocked(uchar*B, size_t Bstep, double y, double x, int width, int height) {
    int closestY = std::round(y);
    int closestX = std::round(x);
    if (closestY < 0 || height <= closestY) return;
    if (closestX < 0 || width <= closestX) return;
    cell<uchar>(B, Bstep, closestY, closestX) = 1;
}

__host__
std::vector<Cand> candidateIterativeSearch(
    const uchar* F, size_t Fstep,
    cv::cuda::GpuMat& gpuF,
    const std::vector<Cand>& tCandidates,
    int width, int height
) {
    cv::Mat BLOCKED(height, width, CV_8U, cv::Scalar(0)); 
    std::vector<Cand> chosenCandidates;
    //
    int n = 1;
    for (const Cand& startCand : tCandidates) {
        candidateExpand(
            F, Fstep,
            gpuF,
            BLOCKED.ptr<uchar>(), BLOCKED.step,
            chosenCandidates,
            startCand, -1, 1.0,
            width, height
        );
        //
        if (n <= 100 || n % 100 == 0) {
            std::cout << "Iterative search: #candidates = " << chosenCandidates.size();
            std::cout << " (" << round(100.0*n/tCandidates.size()) << "%)";
            std::cout << std::endl;
        }
        n++;
    }
    //
    return chosenCandidates;
}

__host__ 
void candidateExpand(
    const uchar *F, size_t Fstep, 
    cv::cuda::GpuMat& gpuF,
    uchar* B, size_t Bstep,
    std::vector<Cand> &chosenCand,
    Cand cand,
    int invEdgeDir,
    double prevScore,
    int width, int height
) {
    if (isBlocked(B, Bstep, cand.y, cand.x, width, height) || cand.score < CAND_THRESHOLD) return;
    //
    if (cand.score < prevScore) {
        cand = upgradeCandidate(F, Fstep, gpuF, cand, width, height);
    }
    //
    if (isBlocked(B, Bstep, cand.y, cand.x, width, height) || cand.score < CAND_THRESHOLD) return;
    // 
    chosenCand.push_back(cand);
    setBlocked(B, Bstep, cand.y, cand.x, width, height);
    //
    int edge1 = getOrthogonalDirection(cand.dir);
    int edge2 = getOppositeDirection(edge1);
    //
    Vec unitEdge1 = getUnitVector(edge1);
    Vec unitEdge2 = getUnitVector(edge2);
    //
    if (edge1 != invEdgeDir) {
        double y1 = cand.y + unitEdge1.y;
        double x1 = cand.x + unitEdge1.x;
        double score1 = computeLabScore(F, Fstep, y1, x1, cand.dir, width, height);
        candidateExpand(
            F, Fstep, gpuF, B, Bstep, chosenCand,
            Cand(y1, x1, cand.dir, score1),
            edge2, cand.score,
            width, height
        );
    }
    //
    if (edge2 != invEdgeDir) {
        double y2 = cand.y + unitEdge2.y;
        double x2 = cand.x + unitEdge2.x;
        double score2 = computeLabScore(F, Fstep, y2, x2, cand.dir, width, height);
        candidateExpand(
            F, Fstep, gpuF, B, Bstep, chosenCand,
            Cand(y2, x2, cand.dir, score2),
            edge1, cand.score, 
            width, height
        );
    }
}

__global__
void bestPixelScoreKernelDirection(
    const uchar* F, size_t Fstep,
    double y, double x,
    double* scores,
    int width, int height
) {
    int dir = threadIdx.x;
    if (dir >= DIRECTIONS) return;
    //
    scores[dir] = computeLabScore(F, Fstep, y, x, dir, width, height);
}

__host__
Cand computeBestPixelCandidate(
    cv::cuda::GpuMat& F,
    double y, double x
) {
    dim3 block(DIRECTIONS); // one thread for every direction;
    dim3 grid(1); // one block for the one pixel
    //
    double* dScores;
    cudaMalloc(&dScores, DIRECTIONS*sizeof(double));
    //
    bestPixelScoreKernelDirection<<<grid, block>>>(
        F.ptr<uchar>(), F.step,
        y, x,
        dScores,
        F.cols, F.rows
    ); 
    cudaDeviceSynchronize();
    //
    double scores[DIRECTIONS];
    cudaMemcpy(
        scores,
        dScores,
        DIRECTIONS*sizeof(double),
        cudaMemcpyDeviceToHost
    );
    cudaFree(dScores);
    //
    int bestDir = 0;
    for (int d = 1; d < DIRECTIONS; d++) {
        if (scores[d] > scores[bestDir]) {
            bestDir = d;
        }
    }
    //
    return Cand(y, x, bestDir, scores[bestDir]);
}
