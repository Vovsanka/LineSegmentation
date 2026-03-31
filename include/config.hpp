#ifndef CONFIG_HPP
#define CONFIG_HPP

// math includes
#include <math.h>
#include <cmath>

// opencv cuda includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thrust/tuple.h>

// constants
constexpr double TOL = 1e-6;
constexpr double INF = 1e6;
constexpr double PI = 3.141593;

// computation parameters
constexpr int DIRECTIONS = 32; // even! <= 1024 // 32k for efficiency  // DIRECTIONS ~ PI

// gray score function
constexpr int G_WINDOW_RADIUS = 3; // Gaussian window size
constexpr double G_SIGMA = 1.0; // standard deviation of the Gaussian
constexpr double EDGE_SHARPNESS = 20000.0;

// beam score function
constexpr int CIRCLE_COUNT = 3;
constexpr double CIRCLE_STEP = 1.8;
constexpr double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise

// threshold candidates
constexpr double CAND_THRESHOLD = 0.5;

// iterative candidates
constexpr double UPPER_THRESHOLD = 0.6; // >= CAND_THRESHOLD
constexpr double LOWER_THRESHOLD = 0.2;
constexpr int UP_COUNT = 10;
constexpr double UP_STEP = 0.1;
constexpr double EXPANSION_STEP = 1.5;
constexpr double EXPANSION_UPGRADE_DELTA = 0.0; // 0 for maximal power of the iterative search

// candidate graph for clustering
constexpr double CONNECTION_RADIUS = 20.0;
constexpr double LINE_THICKNESS = 10.0; 
constexpr double SIMILAR_DIR_ANGLE = 0.12*(PI/2.0);
constexpr double LINE_TRIANGLE_FACTOR = 1.03;
constexpr double COST_BOUND = 10;

// line extraction
constexpr int MIN_LINE_CLUSTER = 12;

#endif
