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

INCDIRS		= -I../../..
CPPFLAGS	= -DNDEBUG -DUSE_TBB
CFLAGS		= -g -O3
NVCCFLAGS	= -g
ifneq ($(findstring icpc,$(CXX)),)
  CFLAGS	= -O3 -g
  NVCCFLAGS	= -O			# must < -O2
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -ltbb
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
EXTHDRS		= ../../../TU/Array++.h \
		../../../TU/Camera++.h \
		../../../TU/Geometry++.h \
		../../../TU/Image++.h \
		../../../TU/Manip.h \
		../../../TU/Minimize.h \
		../../../TU/Profiler.h \
		../../../TU/Quantizer.h \
		../../../TU/Vector++.h \
		../../../TU/WeightedMedianFilter.h \
		../../../TU/algorithm.h \
		../../../TU/functional.h \
		../../../TU/iterator.h \
		../../../TU/pair.h \
		../../../TU/range.h \
		../../../TU/tuple.h \
		../../../TU/types.h
HDRS		=
SRCS		= ImageBase.cc \
		main.cc \
		manipulators.cc
OBJS		= ImageBase.o \
		main.o \
		manipulators.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
ImageBase.o: ../../../TU/Image++.h ../../../TU/types.h \
	../../../TU/Geometry++.h ../../../TU/Vector++.h ../../../TU/Array++.h \
	../../../TU/iterator.h ../../../TU/tuple.h ../../../TU/functional.h \
	../../../TU/range.h ../../../TU/pair.h ../../../TU/Minimize.h \
	../../../TU/Camera++.h ../../../TU/Manip.h
main.o: ../../../TU/Image++.h ../../../TU/types.h ../../../TU/Geometry++.h \
	../../../TU/Vector++.h ../../../TU/Array++.h ../../../TU/iterator.h \
	../../../TU/tuple.h ../../../TU/functional.h ../../../TU/range.h \
	../../../TU/pair.h ../../../TU/Minimize.h \
	../../../TU/WeightedMedianFilter.h ../../../TU/Quantizer.h \
	../../../TU/algorithm.h ../../../TU/Profiler.h
manipulators.o: ../../../TU/Manip.h ../../../TU/types.h
