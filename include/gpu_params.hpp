#ifndef GPU_PARAMS
#define GPU_PARAMS

#include <opencv2/opencv.hpp>

#include "config.hpp"

// GPU params (computed automatically from the constats in config)
inline const dim3 GPU_BLOCK(DIRECTIONS, 1, 1); // one thread for every direction
inline const int GPU_WARP_COUNT = (DIRECTIONS + 31) / 32; // warp for every 32 threads (all warps get accumulated)
inline const size_t GPU_SHMEM_SIZE = GPU_WARP_COUNT * (sizeof(double) + sizeof(int)); // shared memory size

#endif