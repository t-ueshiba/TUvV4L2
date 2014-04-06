#
#  $Id: Makefile,v 1.113 2012-09-15 07:21:08 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DNDEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/FIRFilter.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/IIRFilter.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/SeparableFilter2.h \
		/usr/local/include/TU/Serial.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/fdstream.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/types.h \
		windows/fakeWindows.h
HDRS		= TU/Array++.h \
		TU/BandMatrix++.h \
		TU/Bezier++.h \
		TU/BlockDiagonalMatrix++.h \
		TU/BoxFilter.h \
		TU/Camera++.h \
		TU/CorrectIntensity.h \
		TU/DP.h \
		TU/DericheConvolver.h \
		TU/EdgeDetector.h \
		TU/FIRFilter.h \
		TU/FIRGaussianConvolver.h \
		TU/Filter2.h \
		TU/GaussianConvolver.h \
		TU/Geometry++.h \
		TU/GraphCuts.h \
		TU/GuidedFilter.h \
		TU/Heap.h \
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
		TU/Random.h \
		TU/Ransac.h \
		TU/SHOT602.h \
		TU/SeparableFilter2.h \
		TU/Serial.h \
		TU/SparseMatrix++.h \
		TU/TriggerGenerator.h \
		TU/Vector++.h \
		TU/Warp.h \
		TU/algorithm.h \
		TU/fdstream.h \
		TU/functional.h \
		TU/io.h \
		TU/iterator.h \
		TU/mmInstructions.h \
		TU/tmp.h \
		TU/types.h
SRCS		= BlockDiagonalMatrix++.inst.cc \
		ConversionFromYUV.cc \
		CorrectIntensity.cc \
		EdgeDetector.cc \
		FIRGaussianCoefficients.cc \
		GaussianCoefficients.cc \
		GenericImage.cc \
		Image++.inst.cc \
		ImageBase.cc \
		ImageLine.cc \
		PM16C_04.cc \
		Profiler.cc \
		Random.cc \
		SHOT602.cc \
		Serial.cc \
		TriggerGenerator.cc \
		Vector++.inst.cc \
		Warp.cc \
		fdstream.cc \
		io.cc \
		manipulators.cc
OBJS		= BlockDiagonalMatrix++.inst.o \
		ConversionFromYUV.o \
		CorrectIntensity.o \
		EdgeDetector.o \
		FIRGaussianCoefficients.o \
		GaussianCoefficients.o \
		GenericImage.o \
		Image++.inst.o \
		ImageBase.o \
		ImageLine.o \
		PM16C_04.o \
		Profiler.o \
		Random.o \
		SHOT602.o \
		Serial.o \
		TriggerGenerator.o \
		Vector++.inst.o \
		Warp.o \
		fdstream.o \
		io.o \
		manipulators.o

include $(PROJECT)/lib/l.mk
###
BlockDiagonalMatrix++.inst.o: TU/BlockDiagonalMatrix++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h
ConversionFromYUV.o: TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h
CorrectIntensity.o: TU/CorrectIntensity.h TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h
EdgeDetector.o: TU/EdgeDetector.h TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h
FIRGaussianCoefficients.o: TU/FIRGaussianConvolver.h \
	/usr/local/include/TU/FIRFilter.h \
	/usr/local/include/TU/SeparableFilter2.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h
GaussianCoefficients.o: TU/GaussianConvolver.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/IIRFilter.h \
	/usr/local/include/TU/SeparableFilter2.h \
	/usr/local/include/TU/Minimize.h
GenericImage.o: TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h
Image++.inst.o: TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h
ImageBase.o: TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h TU/Camera++.h TU/Manip.h
ImageLine.o: TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h
PM16C_04.o: TU/PM16C_04.h /usr/local/include/TU/Serial.h \
	/usr/local/include/TU/fdstream.h /usr/local/include/TU/types.h \
	TU/Manip.h
Profiler.o: TU/Profiler.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h
Random.o: TU/Random.h /usr/local/include/TU/types.h windows/fakeWindows.h
SHOT602.o: TU/SHOT602.h /usr/local/include/TU/Serial.h \
	/usr/local/include/TU/fdstream.h /usr/local/include/TU/types.h \
	TU/Manip.h
Serial.o: /usr/local/include/TU/Serial.h /usr/local/include/TU/fdstream.h \
	/usr/local/include/TU/types.h
TriggerGenerator.o: TU/TriggerGenerator.h /usr/local/include/TU/Serial.h \
	/usr/local/include/TU/fdstream.h /usr/local/include/TU/types.h \
	TU/Manip.h
Vector++.inst.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h
Warp.o: TU/Warp.h TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h TU/Camera++.h
fdstream.o: /usr/local/include/TU/fdstream.h /usr/local/include/TU/types.h
io.o: TU/io.h
manipulators.o: TU/Manip.h /usr/local/include/TU/types.h
