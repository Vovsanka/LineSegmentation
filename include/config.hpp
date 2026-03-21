#ifndef CONFIG_HPP
#define CONFIG_HPP

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
constexpr int MAX_SIDE = int(INF); // image scaling down (set infinity to never scale down)
constexpr int DIRECTIONS = 18; // even! // <= 1024 // DIRECTIONS ~ PI //

// gray score function
constexpr int G_WINDOW_RADIUS = 3; // Gaussian window size
constexpr double G_SIGMA = 1.0; // standard deviation of the Gaussian
constexpr double EDGE_SHARPNESS = 700.0;

// beam score function
constexpr int CIRCLE_COUNT = 3;
constexpr double CIRCLE_STEP = 1.0;
constexpr double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise

// threshold candidates
constexpr double CAND_THRESHOLD = 0.5;

// iterative candidates
constexpr double UPPER_THRESHOLD = 0.6; // >= CAND_THRESHOLD
constexpr double LOWER_THRESHOLD = 0.2;
constexpr double EXPANSION_STEP = 1.5;

// candidate upgrade
constexpr int UP_COUNT = 10;
constexpr double UP_STEP = 0.1;

// candidate graph
constexpr double CONNECTION_RADIUS = 10.0;
constexpr double GOOD_DIST_TO_CAND_LINE = 1.5;
constexpr double LINE_TRIANGLE_FACTOR = 1.1;
constexpr double MIN_GAP_SIZE = 5.0;

// clustering cost
constexpr double COST_BOUND = 10;

// line extraction
constexpr int MIN_LINE_CLUSTER = 5;

#endif
