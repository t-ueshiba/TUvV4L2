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
CPPFLAGS	= -DNDEBUG -DPROFILE -DUSE_TBB
CFLAGS		= -O3
NVCCFLAGS	= -g
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DAVX2
  CFLAGS       += -march=core-avx2
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUNewTools++ -ltbb

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
		../../TU/Filter2.h \
		../../TU/Image++.h \
		../../TU/Profiler.h \
		../../TU/Vector++.h \
		../../TU/algorithm.h \
		../../TU/iterator.h \
		../../TU/pair.h \
		../../TU/range.h \
		../../TU/tuple.h \
		../../TU/types.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: ../../TU/Image++.h ../../TU/types.h ../../TU/pair.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/range.h \
	../../TU/algorithm.h ../../TU/tuple.h ../../TU/iterator.h \
	../../TU/BoxFilter.h ../../TU/Filter2.h ../../TU/Profiler.h
