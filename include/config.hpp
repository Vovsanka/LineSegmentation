#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <filesystem>

// path constants
const std::filesystem::path WORKING_STATE_DIR = "../working-state-apb1-18";
const std::filesystem::path IMAGE_PATH = "../images/apb1.png";

// computation parameters
const int MAX_SIDE = 640; // image scaling (up and down possible)
const int DIRECTIONS = 18; // even! // <= 1024 // DIRECTIONS ~ PI //

// constants
const double TOL = 1e-6;
const double INF = 1e6;
const double PI = 3.141593;

// beam score computation
const int CIRCLE_RADIUS = 3;
const double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise

// threshold candidates
const double CAND_THRESHOLD = 0.5;

// iterative candidates
const double UPPER_THRESHOLD = 0.7; // >= CAND_THRESHOLD
const double LOWER_THRESHOLD = 0.3;

// candidate upgrade
const int UP_COUNT = 10;
const double UP_STEP = 0.1;

// candidate graph
const double CONNECTION_RADIUS = 5.0;
const double GOOD_DIST_TO_CAND_LINE = 1.1;
const double LINE_TRIANGLE_FACTOR = 1.1;
const double MIN_GAP_SIZE = 2.5;

// clustering cost
const double COST_BOUND = 10;

// line extraction
const int MIN_LINE_CLUSTER = 5;

#endif
