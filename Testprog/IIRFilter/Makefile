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
CPPFLAGS	= -DNDEBUG #-DUSE_TBB
CFLAGS		= -g -O
NVCCFLAGS	= -g
ifneq ($(findstring icpc,$(CXX)),)
  CFLAGS	= -O3
  NVCCFLAGS	= -O			# must < -O2
#  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUTools++ -ltbb -ltbbmalloc
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
		../../TU/DericheConvolver.h \
		../../TU/GaussianConvolver.h \
		../../TU/Geometry++.h \
		../../TU/IIRFilter.h \
		../../TU/Image++.h \
		../../TU/Minimize.h \
		../../TU/Profiler.h \
		../../TU/SeparableFilter2.h \
		../../TU/Vector++.h \
		../../TU/functional.h \
		../../TU/iterator.h \
		../../TU/mmInstructions.h \
		../../TU/tuple.h \
		../../TU/types.h \
		../../windows/fakeWindows.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: ../../TU/Image++.h ../../TU/types.h ../../TU/Geometry++.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/iterator.h \
	../../TU/tuple.h ../../TU/functional.h ../../TU/mmInstructions.h \
	../../TU/Minimize.h ../../TU/DericheConvolver.h ../../TU/IIRFilter.h \
	../../TU/SeparableFilter2.h ../../TU/GaussianConvolver.h \
	../../TU/Profiler.h ../../windows/fakeWindows.h
