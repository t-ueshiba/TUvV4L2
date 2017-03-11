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

INCDIRS		= -I..
CPPFLAGS	= -D_DEBUG #-DNDEBUG
CFLAGS		= -g
NVCCFLAGS	= -O
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE3
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
EXTHDRS		= ../TU/Array++.h \
		../TU/BlockDiagonalMatrix++.h \
		../TU/Camera++.h \
		../TU/Geometry++.h \
		../TU/Minimize.h \
		../TU/Vector++.h \
		../TU/algorithm.h \
		../TU/iterator.h \
		../TU/range.h \
		../TU/tuple.h
HDRS		= CameraCalibrator.h
SRCS		= CameraCalibrator.cc \
		main.cc
OBJS		= CameraCalibrator.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
CameraCalibrator.o: CameraCalibrator.h ../TU/Geometry++.h ../TU/Vector++.h \
	../TU/Array++.h ../TU/range.h ../TU/algorithm.h ../TU/tuple.h \
	../TU/iterator.h ../TU/Minimize.h ../TU/BlockDiagonalMatrix++.h
main.o: CameraCalibrator.h ../TU/Geometry++.h ../TU/Vector++.h \
	../TU/Array++.h ../TU/range.h ../TU/algorithm.h ../TU/tuple.h \
	../TU/iterator.h ../TU/Minimize.h ../TU/BlockDiagonalMatrix++.h \
	../TU/Camera++.h
