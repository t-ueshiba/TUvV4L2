cmake_minimum_required(VERSION 2.4)
project(TUv++)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_modules")
include(FindX11_Xaw3d)

set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

find_package(X11 REQUIRED)
find_package(X11_Xaw3d REQUIRED)
find_package(Boost REQUIRED)
include_directories(
    .
    ${X11_INCLUDE_DIRS}
    ${Xaw3d_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

add_definitions(-DNDEBUG)

file(GLOB sources *.cc *.c)
file(GLOB headers *.h TU/v/*.h)

add_library(TUv++ SHARED ${sources} ${headers})

target_link_libraries(
    TUv++
    TUTools++
    ${X11_Xaw3d_LIB}
    ${X11_Xt_LIB}
    ${X11_Xmu_LIB}
    ${X11_LIBRARIES})

install(TARGETS TUv++ LIBRARY DESTINATION lib)

add_subdirectory(TU)

