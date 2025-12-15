#ifndef CONFIG_HPP
#define CONFIG_HPP

// constants
const float PI = 3.141593;

// image scaling (up and down possible)
const int MAX_SIDE = 320;

// beam score computation
const int DIRECTIONS = 36;
const int CIRCLE_COUNT = 5;
const float CIRCLE_STEP = 0.5;
const float COLOR_OFFSET = 0.01; // avoid 0-arrays
const float SCORE_BOOSTER = 2.0;

// threshold candidates
const float CAND_THRESHOLD = 0.8;

// candidate upgrade
const int UP_COUNT = 10;
const float UP_STEP = 0.1;
const int UP_ITERATIONS = 5; 

// iterative candidates
const float HIGH_THRESHOLD = 0.9;
const float LOW_THRESHOLD = 0.5;

// iterative search
const float ITER_STEP = 0.5;


#endif
