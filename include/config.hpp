#ifndef CONFIG_HPP
#define CONFIG_HPP

const int DIRECTIONS = 36; // direction pool (30-180 is perfect)
const int SCALE = 2; // resize factor
const double R = 7.0; // radius for the circle of interesting pixels
const double STEP = 1.0; // (R/STEP = INT!) step size for the discrete subpixels in the circle
const double OFFSET = 7; // score offset to reduce the noise sensitivity
// const double LW = 1; // LAB color space: weight of the L channel [0, 1] (A, B channels are weightes as a half of the rest)
const double THICKNESS = 1; // line thickness 
const double CAND_RATIO = 1.8; // candidate ratio
const double THRESHOLD = 0.5; // candidate threshold (the logistic middle)
//
const double UP_STEP = 0.1; // step size for the candidate upgrade
const int UP_COUNT = 14; // sqrt(2)/UP_STEP
const int UP_ITERATIONS = 5; // candidate upgrade iterations
//
const int N = 180; // (N >= 3) N-search
const double DIR_PRECISION = 1e-6; // best direction precision

#endif
