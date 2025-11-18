#ifndef CONFIG_HPP
#define CONFIG_HPP

const int DIRECTIONS = 8;
const double R = 5.0; // radius for the circle of interesting pixels
const double STEP = 0.2; // (R/STEP = INT!) step size for the discrete subpixels in the circle
const double OFFSET = 5; // score offset to reduce the noise sensitivity
const double THICKNESS = 1; // line thickness 
const double CAND_RATIO = 2.0; // candidate ratio
const double THRESHOLD = 0.5; // candidate threshold (the logistic middle)

#endif