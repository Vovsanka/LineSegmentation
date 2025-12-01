#ifndef CONFIG_HPP
#define CONFIG_HPP

const int DIRECTIONS = 18; // direction pool (30-180 is perfect)
const int SCALE = 2; // resize factor
const double R = 10.0; // radius for the circle of interesting pixels
const double STEP = 1.0; // (R/STEP = INT!) step size for the discrete subpixels in the circle
const double OFFSET = 30.0; // score offset to reduce the noise sensitivity
// const double LW = 1; // LAB color space: weight of the L channel [0, 1] (A, B channels are weightes as a half of the rest)
const double THICKNESS = 1.0; // line thickness 
const double SCORE_EXP = 3; // candidate ratio
const double THRESHOLD = 0.8; // candidate threshold (the logistic middle)
//
const double UP_STEP = 0.1; // step size for the candidate upgrade
const int UP_COUNT = 14; // sqrt(2)/UP_STEP
const int UP_ITERATIONS = 5; // candidate upgrade iterations
//
const int N = 180; // (N >= 3) N-search
const double DIR_PRECISION = 1e-6; // best direction precision
//
const double START_THRESHOLD = 0.9;
const double MIN_THRESHOLD = 0.5;

#endif
