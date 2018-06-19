cmake_minimum_required(VERSION 2.4)

project(TUV4L2++)

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

find_package(Boost REQUIRED)

include_directories(. ${Boost_INCLUDE_DIRS})
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

file(GLOB sources *.cc)
file(GLOB headers TU/*.h)

add_library(TUV4L2++ SHARED ${sources} ${headers})

target_link_libraries(TUV4L2++ TUTools++)

install(TARGETS TUV4L2++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
