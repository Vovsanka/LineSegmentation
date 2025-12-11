#ifndef CONFIG_HPP
#define CONFIG_HPP

// image scaling (up and down possible)
const int MAX_SIDE = 300;

// beam score computation
const int DIRECTIONS = 8;
const int CIRCLE_COUNT = 10;
const double CIRCLE_STEP = 0.5;

// threshold candidates
const double CAND_THRESHOLD = 0.8;

// best possible score
const int BEST_N = 1000; 
const double BEST_PRECISION = 1e-6; 

// candidate upgrade
const int UP_COUNT = 10;
const double UP_STEP = 0.1;
const int UP_ITERATIONS = 5; 

// iterative candidates
const double HIGH_THRESHOLD = 0.9;
const double LOW_THRESHOLD = 0.5;

// iterative seach
const double ITER_STEP = 0.5;


#endif
