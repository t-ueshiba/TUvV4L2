#
#  $Id: Makefile,v 1.109 2012-08-16 02:02:36 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/TU
INCDIRS		=

NAME		= $(shell basename $(PWD))

#CCC		= g++
CPPFLAGS	= #-std=c++0x #-DLIBTUTOOLS_DEBUG
CFLAGS		= -g -Wextra -O
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
  else
    CPPFLAGS   += -DSSE3
    CFLAGS     += -xSSE3
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		= TU/BlockDiagonalMatrix++.h \
		TU/Camera++.h \
		TU/CorrectIntensity.h \
		TU/EdgeDetector.h \
		TU/GaussianConvolver.h \
		TU/Image++.h \
		TU/Manip.h \
		TU/PM16C_04.h \
		TU/Profiler.h \
		TU/Random.h \
		TU/SHOT602.h \
		TU/TU/Geometry++.h \
		TU/TU/IIRFilter.h \
		TU/TU/Serial.h \
		TU/TU/TU/Array++.h \
		TU/TU/TU/Minimize.h \
		TU/TU/TU/TU/functional.h \
		TU/TU/TU/TU/types.h \
		TU/TU/TU/fdstream.h \
		TU/TU/TU/iterator.h \
		TU/TU/Vector++.h \
		TU/TriggerGenerator.h \
		TU/Warp.h \
		TU/io.h \
		TU/mmInstructions.h \
		TU/windows/fakeWindows.h
HDRS		= Array++.h \
		Bezier++.h \
		BlockDiagonalMatrix++.h \
		BoxFilter.h \
		Camera++.h \
		CorrectIntensity.h \
		DericheConvolver.h \
		EdgeDetector.h \
		GaussianConvolver.h \
		Geometry++.h \
		GraphCuts.h \
		GuidedFilter.h \
		Heap.h \
		IIRFilter.h \
		Image++.h \
		IntegralImage.h \
		List.h \
		Manip.h \
		Mesh++.h \
		Minimize.h \
		Movie.h \
		NDTree++.h \
		Nurbs++.h \
		PM16C_04.h \
		PSTree.h \
		Profiler.h \
		Random.h \
		Ransac.h \
		SHOT602.h \
		Serial.h \
		SparseMatrix++.h \
		TriggerGenerator.h \
		Vector++.h \
		Warp.h \
		algorithm.h \
		fdstream.h \
		functional.h \
		io.h \
		iterator.h \
		mmInstructions.h \
		types.h
SRCS		= BlockDiagonalMatrix++.inst.cc \
		ConversionFromYUV.cc \
		CorrectIntensity.cc \
		EdgeDetector.cc \
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

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.109 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
BlockDiagonalMatrix++.inst.o: TU/BlockDiagonalMatrix++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h
ConversionFromYUV.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/iterator.h \
	TU/TU/TU/TU/functional.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h TU/TU/TU/Minimize.h
CorrectIntensity.o: TU/CorrectIntensity.h TU/Image++.h TU/TU/Geometry++.h \
	TU/TU/TU/iterator.h TU/TU/TU/TU/functional.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Minimize.h \
	TU/mmInstructions.h
EdgeDetector.o: TU/EdgeDetector.h TU/Image++.h TU/TU/Geometry++.h \
	TU/TU/TU/iterator.h TU/TU/TU/TU/functional.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Minimize.h \
	TU/mmInstructions.h
GaussianCoefficients.o: TU/GaussianConvolver.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/IIRFilter.h \
	TU/mmInstructions.h TU/TU/TU/Minimize.h
GenericImage.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/iterator.h \
	TU/TU/TU/TU/functional.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h TU/TU/TU/Minimize.h
Image++.inst.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/iterator.h \
	TU/TU/TU/TU/functional.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h TU/TU/TU/Minimize.h
ImageBase.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/iterator.h \
	TU/TU/TU/TU/functional.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h TU/TU/TU/Minimize.h TU/Camera++.h TU/Manip.h
ImageLine.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/iterator.h \
	TU/TU/TU/TU/functional.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h TU/TU/TU/Minimize.h
PM16C_04.o: TU/PM16C_04.h TU/TU/Serial.h TU/TU/TU/fdstream.h \
	TU/TU/TU/TU/types.h TU/Manip.h
Profiler.o: TU/Profiler.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h \
	TU/windows/fakeWindows.h
Random.o: TU/Random.h TU/TU/TU/TU/types.h TU/windows/fakeWindows.h
SHOT602.o: TU/SHOT602.h TU/TU/Serial.h TU/TU/TU/fdstream.h \
	TU/TU/TU/TU/types.h TU/Manip.h
Serial.o: TU/TU/Serial.h TU/TU/TU/fdstream.h TU/TU/TU/TU/types.h
TriggerGenerator.o: TU/TriggerGenerator.h TU/TU/Serial.h \
	TU/TU/TU/fdstream.h TU/TU/TU/TU/types.h TU/Manip.h
Vector++.inst.o: TU/TU/Vector++.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h
Warp.o: TU/Warp.h TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/iterator.h \
	TU/TU/TU/TU/functional.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h TU/TU/TU/Minimize.h TU/Camera++.h \
	TU/mmInstructions.h
fdstream.o: TU/TU/TU/fdstream.h TU/TU/TU/TU/types.h
io.o: TU/io.h
manipulators.o: TU/Manip.h TU/TU/TU/TU/types.h
