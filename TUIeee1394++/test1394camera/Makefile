#
#  $Id: Makefile,v 1.3 2002-12-18 04:34:08 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include -I/usr/local/include `gtk-config --cflags`
SRCDIR		= $(HOME)/VVV/Pkg/vvv/utilities/test1394camera

PROGRAM		= test1394camera
LIBS		= -lTUIeee1394++ -lTUTools++ -lraw1394 `gtk-config --libs`

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -O3
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Ieee1394++.h \
		/home/ueshiba/include/TU/Image++.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/types.h
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
REV		= $(shell echo $Revision: 1.3 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
include $(PROJECT)/lib/p-install-srcs.mk
###
My1394Camera.o: My1394Camera.h /home/ueshiba/include/TU/Ieee1394++.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
MyDialog.o: MyDialog.h /home/ueshiba/include/TU/Ieee1394++.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
createCommands.o: My1394Camera.h /home/ueshiba/include/TU/Ieee1394++.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
createMenubar.o: My1394Camera.h /home/ueshiba/include/TU/Ieee1394++.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	MyDialog.h
main.o: My1394Camera.h /home/ueshiba/include/TU/Ieee1394++.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
