#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
#PROGRAM		= $(shell basename $(PWD))
LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I.
CPPFLAGS	= -DNDEBUG
CFLAGS		= -O3
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
EXTHDRS		=
HDRS		= TU/Array++.h \
		TU/BlockDiagonalMatrix++.h \
		TU/Camera++.h \
		TU/Geometry++.h \
		TU/Image++.h \
		TU/Manip.h \
		TU/Minimize.h \
		TU/Vector++.h \
		TU/algorithm.h \
		TU/iterator.h \
		TU/pair.h \
		TU/range.h \
		TU/tuple.h \
		TU/types.h
SRCS		= GenericImage.cc \
		ImageBase.cc \
		manipulators.cc
OBJS		= GenericImage.o \
		ImageBase.o \
		manipulators.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
GenericImage.o: TU/Image++.h TU/types.h TU/pair.h TU/Vector++.h \
	TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h
ImageBase.o: TU/Image++.h TU/types.h TU/pair.h TU/Vector++.h TU/Array++.h \
	TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h TU/Camera++.h \
	TU/Geometry++.h TU/Minimize.h TU/Manip.h
manipulators.o: TU/Manip.h TU/types.h
