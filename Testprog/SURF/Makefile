#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
PROGRAM		= matchTest
#LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I../.. -I$(PREFIX)/include
CPPFLAGS	= -D_DEBUG -DNDEBUG
CFLAGS		= -O3
NVCCFLAGS	= -O
ifeq ($(shell arch),armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE4
  CFLAGS       += -msse4
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUTools++
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
		../../TU/Feature.h \
		../../TU/FeatureMatch.h \
		../../TU/Geometry++.h \
		../../TU/ICIA.h \
		../../TU/IIRFilter.h \
		../../TU/Image++.h \
		../../TU/IntegralImage.h \
		../../TU/Manip.h \
		../../TU/Minimize.h \
		../../TU/Profiler.h \
		../../TU/Ransac.h \
		../../TU/SURFCreator.h \
		../../TU/SeparableFilter2.h \
		../../TU/Vector++.h \
		../../TU/algorithm.h \
		../../TU/iterator.h \
		../../TU/pair.h \
		../../TU/range.h \
		../../TU/tuple.h
HDRS		= MatchImage.h
SRCS		= MatchImage.cc \
		main.cc
OBJS		= MatchImage.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
MatchImage.o: MatchImage.h ../../TU/Image++.h ../../TU/pair.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/range.h \
	../../TU/algorithm.h ../../TU/iterator.h ../../TU/tuple.h \
	../../TU/FeatureMatch.h ../../TU/Geometry++.h ../../TU/Minimize.h \
	../../TU/Ransac.h ../../TU/Manip.h
main.o: ../../TU/FeatureMatch.h ../../TU/Geometry++.h ../../TU/Minimize.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/range.h \
	../../TU/algorithm.h ../../TU/iterator.h ../../TU/tuple.h \
	../../TU/Ransac.h ../../TU/Manip.h ../../TU/SURFCreator.h \
	../../TU/Feature.h ../../TU/IntegralImage.h ../../TU/Image++.h \
	../../TU/pair.h ../../TU/ICIA.h ../../TU/DericheConvolver.h \
	../../TU/IIRFilter.h ../../TU/SeparableFilter2.h ../../TU/Profiler.h \
	MatchImage.h
