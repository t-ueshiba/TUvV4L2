#
#  $Id: Makefile,v 1.4 2002-08-14 05:21:37 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU/v
INCDIRS		= -I$(HOME)/include -I$(OPENWINHOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DUseXaw #-DUseShm
CFLAGS		= -g
CCFLAGS		= -g

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Geometry++.h \
		/home/ueshiba/include/TU/Image++.h \
		/home/ueshiba/include/TU/List++.h \
		/home/ueshiba/include/TU/Manip.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/types.h \
		/home/ueshiba/include/TU/v/CanvasPane.h \
		/home/ueshiba/include/TU/v/CanvasPaneDC.h \
		/home/ueshiba/include/TU/v/CanvasPaneDC3.h \
		/home/ueshiba/include/TU/v/Colormap.h \
		/home/ueshiba/include/TU/v/DC.h \
		/home/ueshiba/include/TU/v/DC3.h \
		/home/ueshiba/include/TU/v/Menu.h \
		/home/ueshiba/include/TU/v/TUv++.h \
		/home/ueshiba/include/TU/v/Widget-Xaw.h \
		/home/ueshiba/include/TU/v/XDC.h \
		TU/v/XglDC.h
HDRS		= XglDC.h
SRCS		= TUXgl++.sa.cc \
		XglDC.cc \
		XglObject.cc
OBJS		= TUXgl++.sa.o \
		XglDC.o \
		XglObject.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.4 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
include $(PROJECT)/lib/RCS.mk
###
TUXgl++.sa.o: TU/v/XglDC.h /home/ueshiba/include/TU/v/CanvasPaneDC3.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC.h \
	/home/ueshiba/include/TU/v/XDC.h /home/ueshiba/include/TU/v/DC.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h \
	/home/ueshiba/include/TU/v/Colormap.h \
	/home/ueshiba/include/TU/v/CanvasPane.h \
	/home/ueshiba/include/TU/v/TUv++.h /home/ueshiba/include/TU/List++.h \
	/home/ueshiba/include/TU/v/Widget-Xaw.h \
	/home/ueshiba/include/TU/v/Menu.h /home/ueshiba/include/TU/v/DC3.h
XglDC.o: TU/v/XglDC.h /home/ueshiba/include/TU/v/CanvasPaneDC3.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC.h \
	/home/ueshiba/include/TU/v/XDC.h /home/ueshiba/include/TU/v/DC.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h \
	/home/ueshiba/include/TU/v/Colormap.h \
	/home/ueshiba/include/TU/v/CanvasPane.h \
	/home/ueshiba/include/TU/v/TUv++.h /home/ueshiba/include/TU/List++.h \
	/home/ueshiba/include/TU/v/Widget-Xaw.h \
	/home/ueshiba/include/TU/v/Menu.h /home/ueshiba/include/TU/v/DC3.h
XglObject.o: TU/v/XglDC.h /home/ueshiba/include/TU/v/CanvasPaneDC3.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC.h \
	/home/ueshiba/include/TU/v/XDC.h /home/ueshiba/include/TU/v/DC.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h \
	/home/ueshiba/include/TU/v/Colormap.h \
	/home/ueshiba/include/TU/v/CanvasPane.h \
	/home/ueshiba/include/TU/v/TUv++.h /home/ueshiba/include/TU/List++.h \
	/home/ueshiba/include/TU/v/Widget-Xaw.h \
	/home/ueshiba/include/TU/v/Menu.h /home/ueshiba/include/TU/v/DC3.h
