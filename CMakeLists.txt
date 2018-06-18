cmake_minimum_required(VERSION 2.4)
project(TUIIDC++)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

find_package(PkgConfig)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)
find_package(Boost REQUIRED)

add_definitions(-DNDEBUG)

include_directories(. ${Boost_INCLUDE_DIRS} ${LIBUSB_INCLUDE_DIRS})
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

file(GLOB sources *.cc)
file(GLOB headers *.h TU/*.h)

add_library(TUIIDC++ SHARED ${sources} ${headers})

if(APPLE)
    add_subdirectory(MacOSX)
    include_directories(MacOSX)
    target_link_libraries(TUIIDC++ TUTools++ raw1394 ${LIBUSB_LIBRARIES})
else()
    pkg_check_modules(LIBRAW1394 REQUIRED libraw1394)
    include_directories(${LIBRAW1394_INCLUDE_DIRS})
    target_link_libraries(TUIIDC++ TUTools++
			  ${LIBRAW1394_LIBRARIES} ${LIBUSB_LIBRARIES})
endif()

install(TARGETS TUIIDC++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
