#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= GFStereo
LIBS		= -lTUStereo++ -lTUTools++ -ltbb -ltbbmalloc
ifeq ($(OSTYPE), darwin)
  LIBS	       +=
endif

CPPFLAGS	= -DNDEBUG -DPROFILE #-DUSE_TBB #-DWITHOUT_CVTDOWN -DRING
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -O3 -xAVX
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DAVX
endif
CCFLAGS		= $(CFLAGS)

LDFLAGS		= -no_pie
LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/BoxFilter.h \
		/usr/local/include/TU/Camera++.h \
		/usr/local/include/TU/GFStereo.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/Rectify.h \
		/usr/local/include/TU/SeparableFilter2.h \
		/usr/local/include/TU/StereoBase.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/Warp.h \
		/usr/local/include/TU/algorithm.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/io.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

include $(PROJECT)/lib/p.mk

###
main.o: /usr/local/include/TU/io.h /usr/local/include/TU/Rectify.h \
	/usr/local/include/TU/Warp.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Camera++.h \
	/usr/local/include/TU/GFStereo.h /usr/local/include/TU/StereoBase.h \
	/usr/local/include/TU/algorithm.h /usr/local/include/TU/BoxFilter.h \
	/usr/local/include/TU/SeparableFilter2.h \
	/usr/local/include/TU/Profiler.h
