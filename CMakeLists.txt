cmake_minimum_required(VERSION 3.0.0)
project(tiny-vo VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fvisibility=hidden")

find_package(OpenCV REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty)

add_executable(main main.cpp)
target_link_libraries(main basalt-headers Ceres::ceres)
