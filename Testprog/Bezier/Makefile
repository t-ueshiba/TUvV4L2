#
#  $Id: Makefile,v 1.14 2012-08-16 01:30:51 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= beziertest
LIBS		= -lTUOgl++ -lTUv++ -lTUTools++ \
		  -lGLU -lGL -lXaw3d -lXt -lXext -lXmu -lX11

CPPFLAGS	=
CFLAGS		= -g -Wextra
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
  else
    CPPFLAGS   += -DSSE3
  endif
endif
CCFLAGS		= $(CFLAGS)
LDFLAGS		=

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Bezier++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/types.h \
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

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.14 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
draw.o: draw.h /usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/List.h /usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/DC3.h \
	/usr/local/include/TU/Bezier++.h
main.o: /usr/local/include/TU/v/App.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/CmdWindow.h \
	/usr/local/include/TU/v/CanvasPane.h draw.h \
	/usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/DC3.h \
	/usr/local/include/TU/Bezier++.h
