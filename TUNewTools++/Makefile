#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
#PROGRAM		= $(shell basename $(PWD))
LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I.
CPPFLAGS	= -DNDEBUG
CFLAGS		= -O3
NVCCFLAGS	= -O
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
#  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		=
LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		=
HDRS		= TU/Array++.h \
		TU/BandMatrix++.h \
		TU/Bezier++.h \
		TU/BlockDiagonalMatrix++.h \
		TU/BoxFilter.h \
		TU/CCSImage.h \
		TU/Camera++.h \
		TU/ComplexImage.h \
		TU/CorrectIntensity.h \
		TU/DP.h \
		TU/DericheConvolver.h \
		TU/EdgeDetector.h \
		TU/FIRFilter.h \
		TU/FIRGaussianConvolver.h \
		TU/Feature.h \
		TU/FeatureMatch.h \
		TU/Filter2.h \
		TU/GaussianConvolver.h \
		TU/Geometry++.h \
		TU/GraphCuts.h \
		TU/GuidedFilter.h \
		TU/Heap.h \
		TU/ICIA.h \
		TU/IIRFilter.h \
		TU/Image++.h \
		TU/IntegralImage.h \
		TU/List.h \
		TU/Manip.h \
		TU/Mesh++.h \
		TU/Minimize.h \
		TU/Movie.h \
		TU/NDTree++.h \
		TU/Nurbs++.h \
		TU/PM16C_04.h \
		TU/PSTree.h \
		TU/Profiler.h \
		TU/Quantizer.h \
		TU/Random.h \
		TU/Ransac.h \
		TU/Rectify.h \
		TU/SADStereo.h \
		TU/SHOT602.h \
		TU/SURFCreator.h \
		TU/SeparableFilter2.h \
		TU/Serial.h \
		TU/SparseMatrix++.h \
		TU/StereoBase.h \
		TU/StereoIterators.h \
		TU/StereoUtility.h \
		TU/TreeFilter.h \
		TU/TriggerGenerator.h \
		TU/Vector++.h \
		TU/Warp.h \
		TU/WeightedMedianFilter.h \
		TU/algorithm.h \
		TU/fdstream.h \
		TU/io.h \
		TU/iterator.h \
		TU/pair.h \
		TU/range.h \
		TU/simd/transform.h \
		TU/tuple.h \
		TU/types.h
SRCS		= ColorConverter.cc \
		EdgeDetector.cc \
		FIRGaussianCoefficients.cc \
		GenericImage.cc \
		ImageBase.cc \
		Random.cc \
		SHOT602.cc \
		Serial.cc \
		TriggerGenerator.cc \
		fdstream.cc \
		io.cc \
		manipulators.cc
OBJS		= ColorConverter.o \
		EdgeDetector.o \
		FIRGaussianCoefficients.o \
		GenericImage.o \
		ImageBase.o \
		Random.o \
		SHOT602.o \
		Serial.o \
		TriggerGenerator.o \
		fdstream.o \
		io.o \
		manipulators.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
ColorConverter.o: TU/Image++.h TU/types.h TU/pair.h TU/Vector++.h \
	TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h
EdgeDetector.o: TU/EdgeDetector.h TU/Image++.h TU/types.h TU/pair.h \
	TU/Vector++.h TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h \
	TU/iterator.h
FIRGaussianCoefficients.o: TU/FIRGaussianConvolver.h TU/FIRFilter.h \
	TU/SeparableFilter2.h TU/Array++.h TU/range.h TU/algorithm.h \
	TU/tuple.h TU/iterator.h
GenericImage.o: TU/Image++.h TU/types.h TU/pair.h TU/Vector++.h \
	TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h
ImageBase.o: TU/Image++.h TU/types.h TU/pair.h TU/Vector++.h TU/Array++.h \
	TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h TU/Camera++.h \
	TU/Geometry++.h TU/Minimize.h TU/Manip.h
Random.o: TU/Random.h TU/types.h
SHOT602.o: TU/SHOT602.h TU/Serial.h TU/fdstream.h TU/types.h TU/Manip.h
Serial.o: TU/Serial.h TU/fdstream.h TU/types.h
TriggerGenerator.o: TU/TriggerGenerator.h TU/Serial.h TU/fdstream.h \
	TU/types.h TU/Manip.h
fdstream.o: TU/fdstream.h TU/types.h
io.o: TU/io.h
manipulators.o: TU/Manip.h TU/types.h
