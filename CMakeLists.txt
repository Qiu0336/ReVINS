cmake_minimum_required(VERSION 3.5)

project(ReVINS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3  -g")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

add_definitions(-DDATA_DIR="${CMAKE_SOURCE_DIR}/")

find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(Ceres REQUIRED)
find_package(OpenCV 3.4 QUIET)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

file(GLOB_RECURSE SRC src/*.h src/*.cc)

add_executable(ReVINS ${SRC})
target_link_libraries(ReVINS
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Boost_LIBRARIES}
    ${CERES_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/DBow3/build/src/libDBoW3.so
    )





