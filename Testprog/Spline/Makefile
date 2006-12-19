#
#  $Id: Makefile,v 1.4 2006-12-19 07:09:24 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include -I$(OGLHOME)/include

PROGRAM		= splinetest
LIBS		= -lTUOgl++ -lTUv++ -lTUTools++ \
		  -lGLU -lGL -lXaw3d -lXt -lXext -lXmu -lX11

CPPFLAGS	= -DUseXaw -DTUArrayPP_RangeCheck
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR) -L$(X11HOME)/lib

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Array++.cc \
		/Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Image++.h \
		/Users/ueshiba/include/TU/List++.h \
		/Users/ueshiba/include/TU/Manip.h \
		/Users/ueshiba/include/TU/Nurbs++.cc \
		/Users/ueshiba/include/TU/Nurbs++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/functions.h \
		/Users/ueshiba/include/TU/types.h \
		/Users/ueshiba/include/TU/v/App.h \
		/Users/ueshiba/include/TU/v/CanvasPane.h \
		/Users/ueshiba/include/TU/v/CanvasPaneDC.h \
		/Users/ueshiba/include/TU/v/CanvasPaneDC3.h \
		/Users/ueshiba/include/TU/v/CmdWindow.h \
		/Users/ueshiba/include/TU/v/Colormap.h \
		/Users/ueshiba/include/TU/v/DC.h \
		/Users/ueshiba/include/TU/v/DC3.h \
		/Users/ueshiba/include/TU/v/Menu.h \
		/Users/ueshiba/include/TU/v/OglDC.h \
		/Users/ueshiba/include/TU/v/TUv++.h \
		/Users/ueshiba/include/TU/v/Widget-Xaw.h \
		/Users/ueshiba/include/TU/v/XDC.h
HDRS		= draw.h
SRCS		= draw.cc \
		main.cc
OBJS		= draw.o \
		main.o

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

include $(PROJECT)/lib/p.mk
include $(PROJECT)/lib/RCS.mk
###
draw.o: draw.h /Users/ueshiba/include/TU/v/OglDC.h \
	/Users/ueshiba/include/TU/v/CanvasPaneDC3.h \
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
	/Users/ueshiba/include/TU/v/Menu.h /Users/ueshiba/include/TU/v/DC3.h \
	/Users/ueshiba/include/TU/Nurbs++.h \
	/Users/ueshiba/include/TU/Array++.cc \
	/Users/ueshiba/include/TU/Nurbs++.cc \
	/Users/ueshiba/include/TU/functions.h
main.o: /Users/ueshiba/include/TU/v/App.h \
	/Users/ueshiba/include/TU/v/TUv++.h \
	/Users/ueshiba/include/TU/v/Colormap.h \
	/Users/ueshiba/include/TU/Image++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/List++.h \
	/Users/ueshiba/include/TU/v/Widget-Xaw.h \
	/Users/ueshiba/include/TU/v/CmdWindow.h \
	/Users/ueshiba/include/TU/v/CanvasPane.h draw.h \
	/Users/ueshiba/include/TU/v/OglDC.h \
	/Users/ueshiba/include/TU/v/CanvasPaneDC3.h \
	/Users/ueshiba/include/TU/v/CanvasPaneDC.h \
	/Users/ueshiba/include/TU/v/XDC.h /Users/ueshiba/include/TU/v/DC.h \
	/Users/ueshiba/include/TU/Manip.h /Users/ueshiba/include/TU/v/Menu.h \
	/Users/ueshiba/include/TU/v/DC3.h /Users/ueshiba/include/TU/Nurbs++.h \
	/Users/ueshiba/include/TU/Array++.cc \
	/Users/ueshiba/include/TU/Nurbs++.cc \
	/Users/ueshiba/include/TU/functions.h
