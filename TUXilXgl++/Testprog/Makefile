#
#  $Id: Makefile,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include -I$(OPENWINHOME)/include

PROGRAM		= xilxgltest
LIBS		= -lTUXilXgl++ -lTUXgl++ -lTUXil++ -lTUv++ -lTUTools++ \
		  -lxgl -lxil -lXaw3d -lXt -lXext -lXmu -lX11 \
		  -ldl -ldga -lthread

CPPFLAGS	= -DUseXaw
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR) -L$(OGLHOME)/lib

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
		/home/ueshiba/include/TU/v/App.h \
		/home/ueshiba/include/TU/v/CanvasPane.h \
		/home/ueshiba/include/TU/v/CanvasPaneDC.h \
		/home/ueshiba/include/TU/v/CanvasPaneDC3.h \
		/home/ueshiba/include/TU/v/CmdPane.h \
		/home/ueshiba/include/TU/v/CmdWindow.h \
		/home/ueshiba/include/TU/v/Colormap.h \
		/home/ueshiba/include/TU/v/DC.h \
		/home/ueshiba/include/TU/v/DC3.h \
		/home/ueshiba/include/TU/v/Menu.h \
		/home/ueshiba/include/TU/v/TUv++.h \
		/home/ueshiba/include/TU/v/Widget-Xaw.h \
		/home/ueshiba/include/TU/v/XDC.h \
		/home/ueshiba/include/TU/v/XglDC.h \
		/home/ueshiba/include/TU/v/XilDC.h \
		/home/ueshiba/include/TU/v/XilXglDC.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.1.1.1 $	|		\
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
main.o: /home/ueshiba/include/TU/v/App.h \
	/home/ueshiba/include/TU/v/TUv++.h \
	/home/ueshiba/include/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/v/Widget-Xaw.h \
	/home/ueshiba/include/TU/v/CmdWindow.h \
	/home/ueshiba/include/TU/v/CmdPane.h \
	/home/ueshiba/include/TU/v/XilXglDC.h \
	/home/ueshiba/include/TU/v/XilDC.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC.h \
	/home/ueshiba/include/TU/v/XDC.h /home/ueshiba/include/TU/v/DC.h \
	/home/ueshiba/include/TU/Manip.h \
	/home/ueshiba/include/TU/v/CanvasPane.h \
	/home/ueshiba/include/TU/v/Menu.h /home/ueshiba/include/TU/v/XglDC.h \
	/home/ueshiba/include/TU/v/CanvasPaneDC3.h \
	/home/ueshiba/include/TU/v/DC3.h
