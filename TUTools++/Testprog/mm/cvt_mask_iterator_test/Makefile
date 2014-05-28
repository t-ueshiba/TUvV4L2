#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I. -I$(PREFIX)/include

PROGRAM		= $(shell basename $(PWD))
LIBS		=
ifneq (,$(findstring darwin,$(OSTYPE)))
  LIBS	       += -framework IOKit -framework CoreFoundation -framework CoreServices
endif

CPPFLAGS	= -DNDEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -g #O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DSSE2
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
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
		TU/types.h
SRCS		= main.cc
OBJS		= main.o

include $(PROJECT)/lib/p.mk

###
main.o: TU/mmInstructions.h TU/iterator.h TU/functional.h
