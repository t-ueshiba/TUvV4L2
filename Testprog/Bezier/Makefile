#
#  $Id: Makefile,v 1.1 2002-07-25 04:36:09 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I../.. -I$(HOME)/include -I$(OGLHOME)/include

PROGRAM		= beziertest
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
EXTHDRS		= ../../TU/Array++.cc \
		../../TU/Bezier++.cc \
		../../TU/Bezier++.h \
		../../TU/Geometry++.h \
		../../TU/Image++.h \
		../../TU/List++.h \
		../../TU/Manip.h \
		../../TU/TU/TU/Array++.h \
		../../TU/TU/TU/TU/types.h \
		../../TU/TU/Vector++.h \
		/home/ueshiba/include/TU/v/App.h \
		/home/ueshiba/include/TU/v/CanvasPane.h \
		/home/ueshiba/include/TU/v/CanvasPaneDC.h \
		/home/ueshiba/include/TU/v/CanvasPaneDC3.h \
		/home/ueshiba/include/TU/v/CmdWindow.h \
		/home/ueshiba/include/TU/v/Colormap.h \
		/home/ueshiba/include/TU/v/DC.h \
		/home/ueshiba/include/TU/v/DC3.h \
		/home/ueshiba/include/TU/v/Menu.h \
		/home/ueshiba/include/TU/v/OglDC.h \
		/home/ueshiba/include/TU/v/TUv++.h \
		/home/ueshiba/include/TU/v/Widget-Xaw.h \
		/home/ueshiba/include/TU/v/XDC.h
HDRS		= draw.h
SRCS		= draw.cc \
		main.cc
OBJS		= draw.o \
		main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.1 $	|		\
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
draw.o: draw.h /home/ueshiba/include/TU/v/OglDC.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC3.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC.h \
	/home/ueshiba/include/TU/v/XDC.h /home/ueshiba/include/TU/v/DC.h \
	../../TU/Geometry++.h ../../TU/TU/Vector++.h ../../TU/TU/TU/Array++.h \
	../../TU/TU/TU/TU/types.h ../../TU/Image++.h ../../TU/Manip.h \
	/home/ueshiba/include/TU/v/Colormap.h \
	/home/ueshiba/include/TU/v/CanvasPane.h \
	/home/ueshiba/include/TU/v/TUv++.h ../../TU/List++.h \
	/home/ueshiba/include/TU/v/Widget-Xaw.h \
	/home/ueshiba/include/TU/v/Menu.h /home/ueshiba/include/TU/v/DC3.h \
	../../TU/Bezier++.h ../../TU/Array++.cc ../../TU/Bezier++.cc
main.o: /home/ueshiba/include/TU/v/App.h \
	/home/ueshiba/include/TU/v/TUv++.h \
	/home/ueshiba/include/TU/v/Colormap.h ../../TU/Image++.h \
	../../TU/TU/Vector++.h ../../TU/TU/TU/Array++.h \
	../../TU/TU/TU/TU/types.h ../../TU/List++.h ../../TU/Geometry++.h \
	/home/ueshiba/include/TU/v/Widget-Xaw.h \
	/home/ueshiba/include/TU/v/CmdWindow.h \
	/home/ueshiba/include/TU/v/CanvasPane.h draw.h \
	/home/ueshiba/include/TU/v/OglDC.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC3.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC.h \
	/home/ueshiba/include/TU/v/XDC.h /home/ueshiba/include/TU/v/DC.h \
	../../TU/Manip.h /home/ueshiba/include/TU/v/Menu.h \
	/home/ueshiba/include/TU/v/DC3.h ../../TU/Bezier++.h \
	../../TU/Array++.cc ../../TU/Bezier++.cc
