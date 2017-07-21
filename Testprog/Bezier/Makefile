#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
PROGRAM		= $(shell basename $(PWD))
#LIBRARY		= lib$(shell basename $(PWD))

IDLDIR		= .
IDLS		=

INCDIRS		= -I../.. -I$(PREFIX)/include
CPPFLAGS	= #-DNDEBUG
CFLAGS		= -g
NVCCFLAGS	= -O
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE4
  CFLAGS       += -msse4
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUOgl++ -lTUv++ -lTUTools++ \
		  -lGLU -lGL -lXaw3d -lXt -lXext -lXmu -lX11
ifneq ($(findstring darwin,$(OSTYPE)),)
  LIBS	       += -framework IOKit -framework CoreFoundation -framework CoreServices
endif
ifeq ($(shell arch), aarch64)
  LIBS	       += -ldrm
endif

LINKER		= $(CXX)

PROGDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
HDRSDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu .idl .hh .so
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../../TU/Array++.h \
		../../TU/Bezier++.h \
		../../TU/Geometry++.h \
		../../TU/Image++.h \
		../../TU/List.h \
		../../TU/Manip.h \
		../../TU/Minimize.h \
		../../TU/Vector++.h \
		../../TU/algorithm.h \
		../../TU/iterator.h \
		../../TU/pair.h \
		../../TU/range.h \
		../../TU/tuple.h \
		/usr/local/include/TU/v/App.h \
		/usr/local/include/TU/v/CanvasPane.h \
		/usr/local/include/TU/v/CanvasPaneDC.h \
		/usr/local/include/TU/v/CanvasPaneDC3.h \
		/usr/local/include/TU/v/CmdWindow.h \
		/usr/local/include/TU/v/Colormap.h \
		/usr/local/include/TU/v/DC.h \
		/usr/local/include/TU/v/DC3.h \
		/usr/local/include/TU/v/Menu.h \
		/usr/local/include/TU/v/OglDC.h \
		/usr/local/include/TU/v/TUv++.h \
		/usr/local/include/TU/v/Widget-Xaw.h \
		/usr/local/include/TU/v/XDC.h
HDRS		= draw.h
SRCS		= draw.cc \
		main.cc
OBJS		= draw.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# modified: CPPFLAGS, LIBS
#include $(PROJECT)/lib/cnoid.mk	# modified: CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# added:    PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
draw.o: draw.h /usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h ../../TU/Geometry++.h \
	../../TU/Minimize.h ../../TU/Vector++.h ../../TU/Array++.h \
	../../TU/range.h ../../TU/iterator.h ../../TU/tuple.h \
	../../TU/algorithm.h ../../TU/Image++.h ../../TU/pair.h \
	../../TU/Manip.h /usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	../../TU/List.h /usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/DC3.h \
	../../TU/Bezier++.h
main.o: /usr/local/include/TU/v/App.h /usr/local/include/TU/v/TUv++.h \
	../../TU/List.h ../../TU/Geometry++.h ../../TU/Minimize.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/range.h \
	../../TU/iterator.h ../../TU/tuple.h ../../TU/algorithm.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/Colormap.h ../../TU/Image++.h ../../TU/pair.h \
	/usr/local/include/TU/v/CmdWindow.h \
	/usr/local/include/TU/v/CanvasPane.h draw.h \
	/usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h ../../TU/Manip.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/DC3.h \
	../../TU/Bezier++.h
