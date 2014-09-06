#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I. -I$(PREFIX)/include

PROGRAM		= $(shell basename $(PWD))
LIBS		= -lTUTools++ -ltbb
ifneq ($(findstring darwin,$(OSTYPE)),)
  LIBS	       += -framework IOKit -framework CoreFoundation -framework CoreServices
endif

CPPFLAGS	= -DNDEBUG -DUSE_TBB
CFLAGS		= -g
NVCCFLAGS	= -g
ifneq ($(findstring icpc,$(CXX)),)
  CFLAGS	= -O3 -xAVX
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
#  CPPFLAGS     += -DAVX
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/DericheConvolver.h \
		/usr/local/include/TU/GaussianConvolver.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/IIRFilter.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/SeparableFilter2.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

include $(PROJECT)/lib/p.mk

###
main.o: /usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/DericheConvolver.h \
	/usr/local/include/TU/IIRFilter.h \
	/usr/local/include/TU/SeparableFilter2.h \
	/usr/local/include/TU/GaussianConvolver.h \
	/usr/local/include/TU/Profiler.h
