#ifndef CONFIG_HPP
#define CONFIG_HPP


// constants
const double TOL = 1e-6;
const double INF = 1e6;
const double PI = 3.141593;

// image scaling (up and down possible)
const int MAX_SIDE = 320;

// beam score computation
const int DIRECTIONS = 180; // even! // <= 1024 // DIRECTIONS ~ PI //
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
const double CONNECTION_RADIUS = 20.0;
const double SAME_LINE_FACTOR = 0.1;
const double LINE_TRIANGLE_FACTOR = 1.1;
const double MIN_GAP_SIZE = 5.0;

// clustering cost
const double MIN_COST = -1;
const double MAX_COST = 10;

// line extraction
const int MIN_LINE_CLUSTER = 5;
const double MAX_DIST_TO_LINE = 2.0;

#endif
