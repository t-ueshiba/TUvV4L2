cmake_minimum_required(VERSION 2.4)
project(TUvV4L2++)

set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -msse4 -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

add_definitions(-DNDEBUG -DSSE4)

set(sources
    createFormatMenu.cc
    featureCmds.cc
    selectROI.cc)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

add_library(TUvV4L2++ SHARED ${sources})

target_link_libraries(TUvV4L2++ TUV4L2++ TUv++)

install(TARGETS TUvV4L2++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
