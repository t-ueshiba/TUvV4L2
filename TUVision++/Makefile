#
#  $Id: Makefile,v 1.29 2012-09-15 05:19:14 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I. -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DTUBrepPP_DEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		=
HDRS		= TU/v/Vision++.h
SRCS		= BrepCanvasPane.cc \
		BrepCmdPane.cc
OBJS		= BrepCanvasPane.o \
		BrepCmdPane.o

include $(PROJECT)/lib/l.mk
###
BrepCanvasPane.o: TU/v/Vision++.h /usr/local/include/TU/Brep/Brep++.h \
	/usr/local/include/TU/Object++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/v/CmdPane.h /usr/local/include/TU/v/CmdWindow.h \
	/usr/local/include/TU/v/TUv++.h /usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/Menu.h
BrepCmdPane.o: TU/v/Vision++.h /usr/local/include/TU/Brep/Brep++.h \
	/usr/local/include/TU/Object++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/v/CmdPane.h /usr/local/include/TU/v/CmdWindow.h \
	/usr/local/include/TU/v/TUv++.h /usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/Menu.h
