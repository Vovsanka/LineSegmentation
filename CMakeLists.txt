cmake_minimum_required(VERSION 3.15)

# Project name and C++ standard
project(LineSegmentation VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory
include_directories(include)

# Add the executable
add_executable(${PROJECT_NAME}
    src/main.cpp
)

# Example: link external libraries (optional)
# target_link_libraries(${PROJECT_NAME} PRIVATE SomeLibrary)
