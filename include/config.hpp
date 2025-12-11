#ifndef CONFIG_HPP
#define CONFIG_HPP

// constants
const float PI = 3.141593;

// image scaling (up and down possible)
const int MAX_SIDE = 1000;

// beam score computation
const int DIRECTIONS = 8;
const int CIRCLE_COUNT = 10;
const float CIRCLE_STEP = 0.5;

// threshold candidates
const float CAND_THRESHOLD = 0.8;

// best possible score
const int BEST_N = 1000; 
const float BEST_PRECISION = 1e-2; 

// candidate upgrade
const int UP_COUNT = 10;
const float UP_STEP = 0.1;
const int UP_ITERATIONS = 5; 

// iterative candidates
const float HIGH_THRESHOLD = 0.9;
const float LOW_THRESHOLD = 0.5;

// iterative seach
const float ITER_STEP = 0.5;


#endif
