#ifndef CONFIG_HPP
#define CONFIG_HPP

const int DIRECTIONS = 60;
const int SCALE = 2; // resize factor
const double R = 3.0; // radius for the circle of interesting pixels
const double STEP = 0.5; // (R/STEP = INT!) step size for the discrete subpixels in the circle
const double OFFSET = 7; // score offset to reduce the noise sensitivity
const double THICKNESS = 1; // line thickness 
const double CAND_RATIO = 1.8; // candidate ratio
const double THRESHOLD = 0.5; // candidate threshold (the logistic middle)

#endif