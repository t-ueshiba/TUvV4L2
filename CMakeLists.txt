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

if(NOT EXISTS "/usr/include/linux/videodev2.h")
    return()
endif()

find_package(PkgConfig REQUIRED)
find_package(Boost REQUIRED)
pkg_check_modules(YAML-CPP REQUIRED yaml-cpp)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_INSTALL_PREFIX}/include
    ${Boost_INCLUDE_DIRS}
    ${YAML-CPP_INCLUDE_DIRS}
)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB sources *.cc)
add_library(${CMAKE_PROJECT_NAME} SHARED ${sources})
target_link_libraries(${CMAKE_PROJECT_NAME} TUTools++ ${YAML-CPP_LIBRARIES})

install(TARGETS ${CMAKE_PROJECT_NAME} LIBRARY DESTINATION lib)

add_subdirectory(TU)
add_subdirectory(testv4l2camera)
