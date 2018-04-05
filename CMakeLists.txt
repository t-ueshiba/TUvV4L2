#
#
#
cmake_minimum_required(VERSION 2.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#include_directories(${Boost_INCLUDE_DIRS} ${XAW3D_INCLUDE_DIRS})

include_directories(
    .
    ../TUIIDC++
    ../TUv++
    ../TUTools++)

set(sources
    createFormatMenu.cc
    featureCmds.cc
    selectROI.cc)

if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif ()

add_library(TUvIIDC++ SHARED ${sources})

set(serial "1.0.0")
set(soserial "1")
set_target_properties(TUvIIDC++	PROPERTIES
		      VERSION	${serial}
		      SOVERSION ${soserial})
target_link_libraries(
    TUvIIDC++
    TUIIDC++
    TUv++)

install(TARGETS TUvIIDC++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
