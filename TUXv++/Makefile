#
#  $Id: Makefile,v 1.3 2002-07-29 00:09:06 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU/v
INCDIRS		= -I$(HOME)/include -I$(X11HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DUseXaw
CFLAGS		= -O -g
CCFLAGS		= -O -g

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Image++.h \
		/Users/ueshiba/include/TU/List++.cc \
		/Users/ueshiba/include/TU/List++.h \
		/Users/ueshiba/include/TU/Manip.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/types.h \
		/Users/ueshiba/include/TU/v/CanvasPane.h \
		/Users/ueshiba/include/TU/v/CanvasPaneDC.h \
		/Users/ueshiba/include/TU/v/Colormap.h \
		/Users/ueshiba/include/TU/v/DC.h \
		/Users/ueshiba/include/TU/v/Menu.h \
		/Users/ueshiba/include/TU/v/ShmDC.h \
		/Users/ueshiba/include/TU/v/TUv++.h \
		/Users/ueshiba/include/TU/v/Widget-Xaw.h \
		/Users/ueshiba/include/TU/v/XDC.h \
		TU/v/XvDC.h
HDRS		= XvDC.h
SRCS		= TUXv++.sa.cc \
		XvDC.cc
OBJS		= TUXv++.sa.o \
		XvDC.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.3 $	|		\
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
TUXv++.sa.o: TU/v/XvDC.h /Users/ueshiba/include/TU/v/ShmDC.h \
	/Users/ueshiba/include/TU/v/CanvasPaneDC.h \
	/Users/ueshiba/include/TU/v/XDC.h /Users/ueshiba/include/TU/v/DC.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Image++.h /Users/ueshiba/include/TU/Manip.h \
	/Users/ueshiba/include/TU/v/Colormap.h \
	/Users/ueshiba/include/TU/v/CanvasPane.h \
	/Users/ueshiba/include/TU/v/TUv++.h \
	/Users/ueshiba/include/TU/List++.h \
	/Users/ueshiba/include/TU/v/Widget-Xaw.h \
	/Users/ueshiba/include/TU/v/Menu.h
XvDC.o: TU/v/XvDC.h /Users/ueshiba/include/TU/v/ShmDC.h \
	/Users/ueshiba/include/TU/v/CanvasPaneDC.h \
	/Users/ueshiba/include/TU/v/XDC.h /Users/ueshiba/include/TU/v/DC.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Image++.h /Users/ueshiba/include/TU/Manip.h \
	/Users/ueshiba/include/TU/v/Colormap.h \
	/Users/ueshiba/include/TU/v/CanvasPane.h \
	/Users/ueshiba/include/TU/v/TUv++.h \
	/Users/ueshiba/include/TU/List++.h \
	/Users/ueshiba/include/TU/v/Widget-Xaw.h \
	/Users/ueshiba/include/TU/v/Menu.h \
	/Users/ueshiba/include/TU/List++.cc
