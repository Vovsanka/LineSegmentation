#include "score.hpp"


__host__ __device__
thrust::tuple<double,double,double> computeStructureTensor( // Jxx, Jyy, Jxy
    const uchar* F,
    size_t Fstep,
    double yPixel, double xPixel,
    int width, int height
)  {
    // Sobel kernels
    const double kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const double ky[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    // Gaussian window parameters
    int R = G_WINDOW_SIZE; 
    double inv2s2 = 1.0 / (2.0 * G_SIGMA * G_SIGMA);
    //
    double Jxx = 0.0;
    double Jyy = 0.0;
    double Jxy = 0.0;
    // Loop over Gaussian window
    for (int dv = -R; dv <= R; dv++) {
        for (int du = -R; du <= R; du++) {
            double y = yPixel + dv;
            double x = xPixel + du;
            // Compute Sobel gradients at (y,x)
            double Ix = 0.0, Iy = 0.0;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    double yy = y + j;
                    double xx = x + i;
                    double val = getGrayColor(F, Fstep, yy, xx, width, height);
                    //
                    Ix += val * kx[j+1][i+1];
                    Iy += val * ky[j+1][i+1];
                }
            }
            // Gaussian weight
            double w = exp(-(du*du + dv*dv) * inv2s2);
            // Accumulate tensor components
            Jxx += w * (Ix * Ix);
            Jyy += w * (Iy * Iy);
            Jxy += w * (Ix * Iy);
        }
    }
    //
    return thrust::make_tuple(Jxx, Jyy, Jxy);
}

__host__ __device__
void shellSort(double* a, int n) {
    // Using Ciura's gap sequence (fast for small arrays)
    int gaps[] = {701, 301, 132, 57, 23, 10, 4, 1};

    for (int g = 0; g < 8; ++g) {
        int gap = gaps[g];
        if (gap >= n) continue;

        for (int i = gap; i < n; ++i) {
            double temp = a[i];
            int j = i;
            while (j >= gap && a[j - gap] > temp) {
                a[j] = a[j - gap];
                j -= gap;
            }
            a[j] = temp;
        }
    }
}


__host__ __device__
double emdRing(const double* arr, int d) { // d in [0, 2*DIRECTIONS)
    int edge1 = getOrthogonalDirection(d);
    int edge2 = getOppositeDirection(edge1);
    if (edge1 > edge2) { // make edge1 < edge2
        int temp = edge1;
        edge1 = edge2;
        edge2 = temp;
    }
    // array sum (in order to normalize)
    double sum = 0;
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        sum += arr[k];
    }
    // prefix sum (normalized values)
    double fixed = (1.0/DIRECTIONS); 
    double prefixSum1[2*DIRECTIONS], prefixSum2[2*DIRECTIONS];
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        double val1, val2;
        if (k == edge1 || k == edge2) {
            val1 = val2 = fixed/2.0; 
        } else if (k < edge1 || edge2 < k) {
            val1 = 0.0;
            val2 = fixed;
        } else { 
            val1 = fixed;
            val2 = 0.0;
        }
        // prefix sums of delta of the normalized array values 
        double arrVal = arr[k]/sum;
        double delta1 = arrVal - val1;
        double delta2 = arrVal - val2;
        if (!k) {
            prefixSum1[k] = delta1;
            prefixSum2[k] = delta2;
        } else {
            prefixSum1[k] = prefixSum1[k - 1] + delta1;
            prefixSum2[k] = prefixSum2[k - 1] + delta2;
        }
    }
    // median computation
    shellSort(prefixSum1, 2*DIRECTIONS);
    shellSort(prefixSum2, 2*DIRECTIONS);
    double med1 = (prefixSum1[DIRECTIONS - 1] + prefixSum1[DIRECTIONS])/2.0;
    double med2 = (prefixSum2[DIRECTIONS - 1] + prefixSum2[DIRECTIONS])/2.0; // ring EMD
    double emd1 = 0.0, emd2 = 0.0;
    for (int k = 0; k < 2*DIRECTIONS; k++) {
        emd1 += fabs(prefixSum1[k] - med1);
        emd2 += fabs(prefixSum2[k] - med2);
    }
    //
    return fmin(emd1, emd2); // the value is bounded by DIRECTIONS/2.0 (uniform <-> one side uniform)
}

__host__ __device__
thrust::tuple<uchar,uchar,uchar> getShiftedColorChannels(
    const uchar* F,
    size_t Fstep,
    double y, double x,
    int d, int c,
    int width, int height
) { // d in [0, 2*DIRECTIONS)
    Vec unitVector = getUnitVector(d);
    //
    double yShifted = y + c*CIRCLE_STEP*unitVector.y;
    double xShifted = x + c*CIRCLE_STEP*unitVector.x;
    //
    return getColorChannels(F, Fstep, yShifted, xShifted, width, height);
}


__host__ __device__
Cand bestPossibleScoreDirection(
    const uchar* F, size_t Fstep,
    double yPixel, double xPixel,
    int width, int height,
    bool beamScore
) { 
    double bestScore;
    if (beamScore) {
        bestScore = computeLabScore(F, Fstep, yPixel, xPixel, 0, width, height);
    } else {
        bestScore = computeGrayScore(F, Fstep, yPixel, xPixel, 0, width, height);
    }
    int bestDir = 0;
    for (int d = 1; d < DIRECTIONS; d++) {
        double score;
        if (beamScore) {
            score = computeLabScore(F, Fstep, yPixel, xPixel, d, width, height);
        } else {
            score = computeGrayScore(F, Fstep, yPixel, xPixel, d, width, height);
        }
        if (score > bestScore) {
            bestScore = score;
            bestDir = d;
        }
    }
    return Cand(yPixel, xPixel, bestDir, bestScore);
}
