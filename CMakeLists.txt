cmake_minimum_required(VERSION 2.4)
project(TUTools++)

set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -msse4 -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

add_definitions(-DNDEBUG -DSSE4)

set(sources
  ColorConverter.cc
  EdgeDetector.cc
  FIRGaussianCoefficients.cc
  GaussianCoefficients.cc
  GenericImage.cc
  PM16C_04.cc
  Rectify.cc
  SHOT602.cc
  SURFCreator.cc
  Serial.cc
  TriggerGenerator.cc
  fdstream.cc
  io.cc
  manipulators.cc)

add_library(TUTools++ SHARED ${sources})

install(TARGETS TUTools++ LIBRARY DESTINATION lib)

add_subdirectory(TU)

