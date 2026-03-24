#include "iterative.hpp"


__host__ __device__
Cand upgradeCandidate(
    const uchar* F, size_t Fstep,
    cv::cuda::GpuMat& gpuF,
    Cand cand,
    int width, int height,
    bool beamScore
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
        double score;
        if (beamScore) {
            score = computeLabScore(F, Fstep, y, x, cand.dir, width, height);
        } else {
            score = computeGrayScore(F, Fstep, y, x, cand.dir, width, height);
        }
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
    Cand bestCand = computeBestPixelCandidate(gpuF, bestY, bestX, beamScore);
    return upgradeCandidate(F, Fstep, gpuF, bestCand, width, height, beamScore);
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
    int width, int height,
    bool beamScore
) {
    cv::Mat BLOCKED(height, width, CV_8U, cv::Scalar(0)); 
    std::vector<Cand> chosenCandidates;
    //
    int n = 0;
    for (const Cand& tCand : tCandidates) {
        //
        if (n++ % 1000 == 0) {
            std::cout << "Iterative search: threshold = " << UPPER_THRESHOLD << ", current = " << tCand.score << std::endl;
        }
        //
        if (tCand.score < UPPER_THRESHOLD) break; // assume tCandidates are already sorted by descending score
        Cand startCand = upgradeCandidate(F, Fstep, gpuF, tCand, width, height, beamScore);
        candidateExpand(
            F, Fstep,
            gpuF,
            BLOCKED.ptr<uchar>(), BLOCKED.step,
            chosenCandidates,
            startCand, -1, -1.0,
            width, height,
            beamScore
        );
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
    int width, int height,
    bool beamScore
) {
    if (cand.score < LOWER_THRESHOLD || isBlocked(B, Bstep, cand.y, cand.x, width, height)) return;
    //
    if (cand.score < prevScore) {
        cand = upgradeCandidate(F, Fstep, gpuF, cand, width, height, beamScore);
        if (isBlocked(B, Bstep, cand.y, cand.x, width, height)) return;
    }
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
        double y1 = cand.y +  EXPANSION_STEP*unitEdge1.y;
        double x1 = cand.x + EXPANSION_STEP*unitEdge1.x;
        double score1;
        if (beamScore) {
            score1 = computeLabScore(F, Fstep, y1, x1, cand.dir, width, height);
        } else {
            score1 = computeGrayScore(F, Fstep, y1, x1, cand.dir, width, height);
        }
        candidateExpand(
            F, Fstep, gpuF, B, Bstep, chosenCand,
            Cand(y1, x1, cand.dir, score1),
            edge2, cand.score,
            width, height,
            beamScore
        );
    }
    //
    if (edge2 != invEdgeDir) {
        double y2 = cand.y + EXPANSION_STEP*unitEdge2.y;
        double x2 = cand.x + EXPANSION_STEP*unitEdge2.x;
        double score2;
        if (beamScore) {
            score2 = computeLabScore(F, Fstep, y2, x2, cand.dir, width, height);
        } else {
            score2 = computeGrayScore(F, Fstep, y2, x2, cand.dir, width, height);
        }
        candidateExpand(
            F, Fstep, gpuF, B, Bstep, chosenCand,
            Cand(y2, x2, cand.dir, score2),
            edge1, cand.score, 
            width, height,
            beamScore
        );
    }
}
__global__
void bestPixelScoreKernelDirection(
    const uchar* F, size_t Fstep,
    double y, double x,
    double* scores,
    int width, int height,
    bool beamScore
) {
    int dir = threadIdx.x;
    if (dir >= DIRECTIONS) return;
    //
    if (beamScore) {
        scores[dir] = computeLabScore(F, Fstep, y, x, dir, width, height);
    } else {
        scores[dir] = computeGrayScore(F, Fstep, y, x, dir, width, height);
    }
}

__host__
Cand computeBestPixelCandidate(
    cv::cuda::GpuMat& F,
    double y, double x,
    bool beamScore
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
        F.cols, F.rows,
        beamScore
    ); 
    cudaError_t err = cudaGetLastError(); 
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
