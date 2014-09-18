#
#  $Id: Makefile,v 1.29 2012-09-15 07:21:14 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I. -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DNDEBUG
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
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/v/CanvasPane.h \
		/usr/local/include/TU/v/CanvasPaneDC.h \
		/usr/local/include/TU/v/Colormap.h \
		/usr/local/include/TU/v/DC.h \
		/usr/local/include/TU/v/Menu.h \
		/usr/local/include/TU/v/ShmDC.h \
		/usr/local/include/TU/v/TUv++.h \
		/usr/local/include/TU/v/Widget-Xaw.h \
		/usr/local/include/TU/v/XDC.h
HDRS		= TU/v/XvDC.h
SRCS		= TUXv++.sa.cc \
		XvDC.cc
OBJS		= TUXv++.sa.o \
		XvDC.o

include $(PROJECT)/lib/l.mk
###
TUXv++.sa.o: TU/v/XvDC.h /usr/local/include/TU/v/ShmDC.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/List.h /usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/Menu.h
XvDC.o: TU/v/XvDC.h /usr/local/include/TU/v/ShmDC.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/List.h /usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/Menu.h
