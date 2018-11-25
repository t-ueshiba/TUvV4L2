cmake_minimum_required(VERSION 2.4)
project(TUTools++)

enable_language(CXX)
set(CMAKE_CXX_STANDARD		14)
set(CMAKE_CXX_STANDARD_REQUIRED	ON)
set(CMAKE_CXX_EXTENSIONS	OFF)
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)
add_definitions(-DNDEBUG)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_modules")
include(FindSSE)
FindSSE()

if(SSE3_FOUND)
  if(SSSE3_FOUND)
    set(CPP_DFLAGS -DSSE3)
    set(CXX_DFLAGS "-msse3 -mssse3")
  endif()
endif()
if(SSE4_2_FOUND)
  set(CPP_DFLAGS -DSSE4)
  set(CXX_DFLAGS "-msse4.2 -mpopcnt")
endif()
if(AVX_FOUND)
  set(CPP_DFLAGS -DAVX)
  set(CXX_DFLAGS -mavx)
endif()
if(AVX2_FOUND)
  set(CPP_DFLAGS -DAVX2)
  set(CXX_DFLAGS -mavx2)
endif()

find_package(Boost REQUIRED)
include_directories(${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})

file(GLOB sources *.cc)
add_library(${PROJECT_NAME} SHARED ${sources})

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

add_subdirectory(TU)
add_subdirectory(Testprog)

