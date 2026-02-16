#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <filesystem>

// path constants
const std::filesystem::path WORKING_STATE_DIR = "../working-state-table-90";
const std::filesystem::path IMAGE_PATH = "../images/table.png";

// computation parameters
constexpr int MAX_SIDE = 640; // image scaling (up and down possible)
constexpr int DIRECTIONS = 90; // even! // <= 1024 // DIRECTIONS ~ PI //

// constants
constexpr double TOL = 1e-6;
constexpr double INF = 1e6;
constexpr double PI = 3.141593;

// beam score computation
constexpr int CIRCLE_COUNT = 1;
constexpr double CIRCLE_STEP = 3.0;
constexpr double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise

// threshold candidates
constexpr double CAND_THRESHOLD = 0.5;

// iterative candidates
constexpr double UPPER_THRESHOLD = 0.6; // >= CAND_THRESHOLD
constexpr double LOWER_THRESHOLD = 0.2;

// candidate upgrade
constexpr int UP_COUNT = 10;
constexpr double UP_STEP = 0.1;

// candidate graph
constexpr double CONNECTION_RADIUS = 5.0;
constexpr double GOOD_DIST_TO_CAND_LINE = 0.8;
constexpr double LINE_TRIANGLE_FACTOR = 1.1;
constexpr double MIN_GAP_SIZE = 2.5;

// clustering cost
constexpr double COST_BOUND = 10;

// line extraction
constexpr int MIN_LINE_CLUSTER = 5;

#endif
