cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

project(LineSegmentation LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

find_package(OpenCV REQUIRED)

set(SOURCES
    src/main.cu
    src/color.cu
    src/score.cu
    src/show.cu
    src/iterative.cu
    src/best.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


