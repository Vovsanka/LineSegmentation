#ifndef CONFIG_HPP
#define CONFIG_HPP


// math constants
const double PI = 3.141593;

// image scaling (up and down possible)
const int MAX_SIDE = 320;

// beam score computation
const int DIRECTIONS = 36; // even! // <= 1024 // DIRECTIONS ~ PI //
const int CIRCLE_COUNT = 5; // 5
const double CIRCLE_STEP = 1.0;
const double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise
const double SCORE_BOOSTER = 4.0;

// threshold candidates
const double CAND_THRESHOLD = 0.8;

// candidate upgrade
const int UP_COUNT = 10;
const double UP_STEP = 0.1;

// candidate clustering cost
const double ALMOST_SAME_PIXEL = 1.5; 
const int ALMOST_SAME_DIR = 1;
const double ALMOST_SAME_LINE = 2.0;
const double ALMOST_LINE_TRIANGLE = 2.0;
const double GAP_SIZE = 5.0;

#endif
