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

INCDIRS		= -I../..
CPPFLAGS	= -DNDEBUG
CFLAGS		= -O
NVCCFLAGS	= -g
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE4
  CFLAGS       += -msse4
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
EXTHDRS		= ../../TU/Array++.h \
		../../TU/Profiler.h \
		../../TU/Quantizer.h \
		../../TU/Vector++.h \
		../../TU/WeightedMedianFilter.h \
		../../TU/algorithm.h \
		../../TU/iterator.h \
		../../TU/range.h \
		../../TU/tuple.h \
		../../TU/type_traits.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: ../../TU/WeightedMedianFilter.h ../../TU/Quantizer.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/range.h \
	../../TU/iterator.h ../../TU/tuple.h ../../TU/type_traits.h \
	../../TU/algorithm.h ../../TU/Profiler.h
