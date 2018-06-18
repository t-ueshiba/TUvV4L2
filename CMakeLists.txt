cmake_minimum_required(VERSION 2.4)
project(TUV4L2++)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

find_package(Boost REQUIRED)

add_definitions(-DNDEBUG)

include_directories(
    .
    ${CMAKE_INSTALL_PREFIX}/include
    ${Boost_INCLUDE_DIRS}
)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB sources *.cc)
file(GLOB headers TU/*.h)

add_library(TUV4L2++ SHARED ${sources} ${headers})

target_link_libraries(TUV4L2++ TUTools++)

install(TARGETS TUV4L2++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
