cmake_minimum_required(VERSION 2.4)
project(TUTools++)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_modules")
include(FindSSE)
FindSSE()

if(SSE3_FOUND)
  if(SSSE3_FOUND)
    set(CPP_DFLAGS -DSSE3)
    set(CXX_DFLAGS -msse3 -mssse3)
  endif()
endif()
if(SSE4_2_FOUND)
  set(CPP_DFLAGS -DSSE4)
  set(CXX_DFLAGS -msse4.2 -mpopcnt)
endif()
if(AVX_FOUND)
  set(CPP_DFLAGS -DAVX)
  set(CXX_DFLAGS -mavx)
endif()
if(AVX2_FOUND)
  set(CPP_DFLAGS -DAVX2)
  set(CXX_DFLAGS -mavx2)
endif()

set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} ${CXX_DFLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

find_package(Boost REQUIRED)
include_directories(. ${Boost_INCLUDE_DIRS})

add_definitions(-DNDEBUG ${CPP_DFLAGS})

file(GLOB sources *.cc)
file(GLOB headers TU/*.h TU/simd/*.h TU/simd/x86/*.h)

add_library(TUTools++ SHARED ${sources} ${headers})

install(TARGETS TUTools++ LIBRARY DESTINATION lib)

add_subdirectory(TU)

