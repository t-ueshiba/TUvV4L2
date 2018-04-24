#
#
#
cmake_minimum_required(VERSION 2.4)

set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -msse4 -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

add_definitions(-DNDEBUG -DSSE4)

set(sources
    V4L2Camera.cc
    V4L2CameraArray.cc)

if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif ()

add_library(TUV4L2++ SHARED ${sources})

include_directories(${LIBRAW1394_INCLUDE_DIRS})
target_link_libraries(TUTools++)

install(TARGETS TUV4L2++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
