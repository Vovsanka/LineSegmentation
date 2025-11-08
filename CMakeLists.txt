cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

project(LineSegmentation VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cu
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
