# Line Segmentation App

**Line Segment Identification in Color Images
by Concurrent Edge Detection and Clique Partitioning**

Master's Thesis by Volodymyr Drobitko, Computer Science, Technical University of Dresden.

[![License](https://img.shields.io/badge/license-academic%20use%20only-red)]()
[![C++](https://img.shields.io/badge/C++-11/17-blue)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-C++-5C3EE8)]()
[![CUDA](https://img.shields.io/badge/CUDA-supported-green)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)]()

- The whole APB dataset is stored in the `./APB-dataset` folder.

- Experiment results for Wireframe, YUD+, and APB are stored in the corresponding `./experiments-python/...-analysis-...` folders.

- Tunable parameters can be found in `./include/config.hpp`

## Requirements

- C++ 17
- CMake 3.18
- OpenCV
- CUDA
- Cairo
- Python 3.12 (for experiments)
- pip (for experiments)
- numpy (for experiments)
- pandas (for experiments)
- scipy (for experiments)

## Installation on Linux (Ubuntu)

### Build OpenCV with CUDA support:

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D BUILD_opencv_cudacodec=OFF \
      -D WITH_GTK=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      ..

make -j$(nproc)
sudo make install
```

### Build Line Segmentation App:

```bash
git clone https://github.com/Vovsanka/LineSegmentation.git
cd LineSegmentation

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

./LineSegmentation
```

## Experiments on Linux (Ubuntu)

### Python Environment Setup

```bash
cd experiments-python

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

### Run Experiment Scripts

GOOD parameter configuration (already set):

```bash
   bash run-apb-quality.sh

   bash run-yud-candidates-quality.sh
   bash run-yud-clustering-quality.sh
   bash run-yud-analysis-quality.sh


   bash run-wireframe-candidates-quality.sh
   bash run-wireframe-clustering-quality.sh
   bash run-wireframe-analysis-quality.sh
```

SIMPLE parameter configuration (set in `./include/config.hpp`!): 

```bash
   bash run-yud-candidates-quantity.sh
   bash run-yud-clustering-quantity.sh
   bash run-yud-analysis-quantity.sh

   bash run-wireframe-candidates-quantity.sh
   bash run-wireframe-clustering-quantity.sh
   bash run-wireframe-analysis-quantity.sh
```

SIMPLE parameter configuration: 

```C++ 
#ifndef CONFIG_HPP
#define CONFIG_HPP

// math includes
#include <math.h>
#include <cmath>

// opencv cuda includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thrust/tuple.h>

// constants
constexpr double TOL = 1e-6;
constexpr double INF = 1e6;
constexpr double PI = 3.141593;

// computation parameters
constexpr int DIRECTIONS = 64; // even! <= 1024 // 32k for efficiency  // DIRECTIONS ~ PI

// gray score function
constexpr int G_WINDOW_RADIUS = 3; // Gaussian window size
constexpr double G_SIGMA = 1.0; // standard deviation of the Gaussian
constexpr double EDGE_SHARPNESS = 20000.0;

// beam score function
constexpr int CIRCLE_COUNT = 3;
constexpr double CIRCLE_STEP = 1.0;
constexpr double COLOR_OFFSET = 3.0; // avoid 0-arrays & ignore some noise

// threshold candidates
constexpr double CAND_THRESHOLD = 0.5;

// iterative candidates
constexpr double UPPER_THRESHOLD = 0.6; // >= CAND_THRESHOLD
constexpr double LOWER_THRESHOLD = 0.2;
constexpr int UP_COUNT = 10;
constexpr double UP_STEP = 0.1;
constexpr double EXPANSION_STEP = 1.5;
constexpr double EXPANSION_UPGRADE_DELTA = 0.2; // 0 for maximal power of the iterative search

// candidate graph for clustering
constexpr double CONNECTION_RADIUS = 10.0;
constexpr double LINE_THICKNESS = 8.0; 
constexpr double SIMILAR_DIR_ANGLE = 0.1*(PI/2.0);
constexpr double LINE_TRIANGLE_FACTOR = 1.05;
constexpr double COST_BOUND = 10;

// line extraction
constexpr int MIN_LINE_CLUSTER = 10;

#endif
```


