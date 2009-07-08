#
#  $Id: Makefile,v 1.12 2009-07-08 06:37:32 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(LOCALDIR)/include -I..

PROGRAM		= test1394camera
#LIBS		= -lTUIeee1394++ -lTUTools++ -lraw1394 $(shell gtk-config --libs)
LIBS		= -lTUIeee1394++ -lTUTools++ -lraw1394 $(shell pkg-config gtk+-2.0 --libs)
ifeq ($(OSTYPE), darwin)
  LIBS	       += -framework IOKit -framework CoreFoundation
endif

CPPFLAGS	= #-DUSE_RAWISO
CFLAGS		= -O
#CCFLAGS		= -O $(shell gtk-config --cflags)
CCFLAGS		= -O $(shell pkg-config gtk+-2.0 --cflags)
#CCFLAGS	= -O3 -parallel
LDFLAGS		= $(CCFLAGS) -L$(LIBDIR)
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Ieee1394++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Normalize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/types.h
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
REV		= $(shell echo $Revision: 1.12 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
###
My1394Camera.o: My1394Camera.h /usr/local/include/TU/Ieee1394++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Normalize.h
MyDialog.o: MyDialog.h /usr/local/include/TU/Ieee1394++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Normalize.h
createCommands.o: My1394Camera.h /usr/local/include/TU/Ieee1394++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Normalize.h
createMenubar.o: My1394Camera.h /usr/local/include/TU/Ieee1394++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Normalize.h \
	MyDialog.h
main.o: My1394Camera.h /usr/local/include/TU/Ieee1394++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Normalize.h
