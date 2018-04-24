cmake_minimum_required(VERSION 2.4)
project(TUvIIDC++)

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

add_library(TUvIIDC++ SHARED ${sources})

target_link_libraries(TUvIIDC++ TUIIDC++ TUv++)

install(TARGETS TUvIIDC++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
