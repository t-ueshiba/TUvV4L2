#
#
#
cmake_minimum_required(VERSION 2.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

include_directories(.)

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

set(serial "1.0.0")
set(soserial "1")
set_target_properties(TUTools++ PROPERTIES
		      VERSION	${serial}
		      SOVERSION ${soserial})

install(TARGETS TUTools++ LIBRARY DESTINATION lib)

add_subdirectory(TU)

