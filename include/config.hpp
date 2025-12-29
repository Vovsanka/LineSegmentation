#ifndef CONFIG_HPP
#define CONFIG_HPP

// constants
const double PI = 3.141593;

// image scaling (up and down possible)
const int MAX_SIDE = 640;

// beam score computation
const int DIRECTIONS = 8; // even! // DIRECTIONS ~ PI // 36
const int CIRCLE_COUNT = 5; // 5
const double CIRCLE_STEP = 1.0;
const double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise
const double SCORE_BOOSTER = 4.0;

// threshold candidates
const double CAND_THRESHOLD = 0.8;

// iterative candidates
const double HIGH_THRESHOLD = 0.9;
const double LOW_THRESHOLD = 0.8;

// candidate upgrade
const int UP_COUNT = 100;
const double UP_STEP = 0.01;
const int UP_ITERATIONS = 5; 

// iterative search
const double ITER_STEP = 1.0;


#endif
