#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
PROGRAM		= $(shell basename $(PWD))
#LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I../.. -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG -DPROFILE -DAVX2 #-DUSE_TBB #-DWITHOUT_CVTDOWN
CFLAGS		= -O3
NVCCFLAGS	= -O
ifneq ($(findstring icpc,$(CXX)),)
  CFLAGS       += -xCORE-AVX2
else
  CFLAGS       += -mavx2
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUTools++ -ltbb
ifneq ($(findstring darwin,$(OSTYPE)),)
  LIBS	       += -framework IOKit -framework CoreFoundation \
		  -framework CoreServices
endif

LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../../TU/Array++.h \
		../../TU/BoxFilter.h \
		../../TU/Camera++.h \
		../../TU/GFStereo.h \
		../../TU/Geometry++.h \
		../../TU/Image++.h \
		../../TU/Minimize.h \
		../../TU/Profiler.h \
		../../TU/Rectify.h \
		../../TU/SADStereo.h \
		../../TU/SeparableFilter2.h \
		../../TU/StereoBase.h \
		../../TU/Vector++.h \
		../../TU/Warp.h \
		../../TU/algorithm.h \
		../../TU/functional.h \
		../../TU/io.h \
		../../TU/iterator.h \
		../../TU/mmInstructions.h \
		../../TU/tuple.h \
		../../TU/types.h \
		../../windows/fakeWindows.h
HDRS		=
SRCS		= Rectify.cc \
		main.cc
OBJS		= Rectify.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
Rectify.o: ../../TU/Rectify.h ../../TU/Warp.h ../../TU/Image++.h \
	../../TU/types.h ../../TU/Geometry++.h ../../TU/Vector++.h \
	../../TU/Array++.h ../../TU/iterator.h ../../TU/tuple.h \
	../../TU/functional.h ../../TU/mmInstructions.h ../../TU/Minimize.h \
	../../TU/Camera++.h ../../TU/algorithm.h
main.o: ../../TU/io.h ../../TU/Rectify.h ../../TU/Warp.h \
	../../TU/Image++.h ../../TU/types.h ../../TU/Geometry++.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/iterator.h \
	../../TU/tuple.h ../../TU/functional.h ../../TU/mmInstructions.h \
	../../TU/Minimize.h ../../TU/Camera++.h ../../TU/SADStereo.h \
	../../TU/StereoBase.h ../../TU/algorithm.h ../../TU/Profiler.h \
	../../windows/fakeWindows.h ../../TU/BoxFilter.h \
	../../TU/SeparableFilter2.h ../../TU/GFStereo.h
