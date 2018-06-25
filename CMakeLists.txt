cmake_minimum_required(VERSION 2.4)

project(TUvV4L2++)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD		14)
set(CMAKE_CXX_STANDARD_REQUIRED	ON)
set(CMAKE_CXX_EXTENSIONS	OFF)
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)
add_definitions(-DNDEBUG)

file(GLOB sources *.cc)

add_library(TUvV4L2++ SHARED ${sources})

target_link_libraries(TUvV4L2++ TUV4L2++ TUv++)

install(TARGETS TUvV4L2++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
