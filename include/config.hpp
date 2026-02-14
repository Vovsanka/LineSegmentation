#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <filesystem>

// working state directory path
const std::filesystem::path WORKING_STATE_DIR = "../working-state-black-16";

// constants
const double TOL = 1e-6;
const double INF = 1e6;
const double PI = 3.141593;

// image scaling (up and down possible)
const int MAX_SIDE = 320;

// beam score computation
const int DIRECTIONS = 16; // even! // <= 1024 // DIRECTIONS ~ PI //
const int CIRCLE_COUNT = 5; // 5
const double CIRCLE_STEP = 1.0;
const double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise
const double SCORE_BOOSTER = 4.0;

// threshold candidates
const double CAND_THRESHOLD = 0.8;

// candidate upgrade
const int UP_COUNT = 10;
const double UP_STEP = 0.1;

// candidate graph
const double CONNECTION_RADIUS = 5.0;
const double GOOD_DIST_TO_CAND_LINE = 0.8;
const double LINE_TRIANGLE_FACTOR = 1.1;
const double MIN_GAP_SIZE = 2.5;

// clustering cost
const double MIN_COST = -10;
const double MAX_COST = 10;

// line extraction
const int MIN_LINE_CLUSTER = 5;
const double MAX_DIST_TO_FITTED_LINE = 10;

#endif
