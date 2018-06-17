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

INCDIRS		= -I. -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG
CFLAGS		= -O3 -Wall
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

OTHER_DIR	= $(HOME)/projects/HRP-5P/hrp5p-calib/libraries/TUIIDC++

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Camera++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/algorithm.h \
		/usr/local/include/TU/io.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/range.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/type_traits.h
HDRS		= FireWireNode_.h \
		TU/IIDC++.h \
		TU/IIDCCameraArray.h \
		USBNode_.h
SRCS		= FireWireNode.cc \
		IIDCCamera.cc \
		IIDCCameraArray.cc \
		IIDCNode.cc \
		USBNode.cc
OBJS		= FireWireNode.o \
		IIDCCamera.o \
		IIDCCameraArray.o \
		IIDCNode.o \
		USBNode.o

OTHER_HDRS	= $(HDRS)
OTHER_SRCS	= $(SRCS)

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
include $(PROJECT)/lib/other.mk
###
FireWireNode.o: FireWireNode_.h TU/IIDC++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/type_traits.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/Camera++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/range.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/algorithm.h
IIDCCamera.o: FireWireNode_.h TU/IIDC++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/type_traits.h \
	/usr/local/include/TU/Manip.h /usr/local/include/TU/Camera++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/range.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/algorithm.h \
	USBNode_.h
IIDCCameraArray.o: TU/IIDCCameraArray.h TU/IIDC++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/type_traits.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/Camera++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/range.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/algorithm.h /usr/local/include/TU/io.h
IIDCNode.o: TU/IIDC++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/type_traits.h \
	/usr/local/include/TU/Manip.h /usr/local/include/TU/Camera++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/range.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/algorithm.h
USBNode.o: USBNode_.h TU/IIDC++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/type_traits.h \
	/usr/local/include/TU/Manip.h /usr/local/include/TU/Camera++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/range.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/algorithm.h
