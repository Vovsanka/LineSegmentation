#ifndef CONFIG_HPP
#define CONFIG_HPP


// CUDA constants
const dim3 GPU_BLOCK(16, 16); // one thread for every pixel

// math constants
const double PI = 3.141593;

// image scaling (up and down possible)
const int MAX_SIDE = 160;

// beam score computation
const int DIRECTIONS = 90; // even! <= 1024 // DIRECTIONS ~ PI //
const int CIRCLE_COUNT = 5; // 5
const double CIRCLE_STEP = 1.0;
const double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise
const double SCORE_BOOSTER = 4.0;

// threshold candidates
const double CAND_THRESHOLD = 0.8;

// candidate upgrade
const int UP_COUNT = 10;
const double UP_STEP = 0.1;
const int UP_ITERATIONS = 3; 


#endif
