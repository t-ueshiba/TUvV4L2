#
#  $Id: Makefile,v 1.8 2008-10-16 00:52:49 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I/usr/local/include -I..

PROGRAM		= test1394camera
#LIBS		= -lTUIeee1394++ -lTUTools++ -lraw1394 $(shell gtk-config --libs)
LIBS		= -lTUIeee1394++ -lTUTools++ -lraw1394 $(shell pkg-config --libs gtk+-2.0)
ifeq ($(OSTYPE), darwin)
  LIBS	       += -framework IOKit -framework CoreFoundation
endif

CPPFLAGS	= #-DRAWISO
CFLAGS		= -O
#CCFLAGS		= -O $(shell gtk-config --cflags)
CCFLAGS		= -O $(shell pkg-config --cflags gtk+-2.0)
#CCFLAGS	= -O3 -parallel
LDFLAGS		= $(CCFLAGS) -L$(LIBDIR)
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= ../TU/Ieee1394++.h
HDRS		= My1394Camera.h \
		MyDialog.h
SRCS		= My1394Camera.cc \
		MyDialog.cc \
		createCommands.cc \
		createMenubar.cc \
		main.cc
OBJS		= My1394Camera.o \
		MyDialog.o \
		createCommands.o \
		createMenubar.o \
		main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.8 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
###
My1394Camera.o: My1394Camera.h ../TU/Ieee1394++.h
MyDialog.o: MyDialog.h ../TU/Ieee1394++.h
createCommands.o: My1394Camera.h ../TU/Ieee1394++.h
createMenubar.o: My1394Camera.h ../TU/Ieee1394++.h MyDialog.h
main.o: My1394Camera.h ../TU/Ieee1394++.h
