#
#  $Id: Makefile,v 1.12 2006-12-19 07:05:20 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include -I/usr/local/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DHAVE_TUToolsPP #-DRAWISO
CFLAGS		= -O -g
CCFLAGS		= -g
ifeq ($(CCC), icpc)
  CCFLAGS	= -O3 -parallel
#  CCFLAGS	= -g
endif
LDFLAGS		=
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Array++.cc \
		/Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Image++.cc \
		/Users/ueshiba/include/TU/Image++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/functions.h \
		/Users/ueshiba/include/TU/mmInstructions.h \
		/Users/ueshiba/include/TU/types.h
HDRS		= Ieee1394++.h
SRCS		= Ieee1394Camera.cc \
		Ieee1394Node.cc
OBJS		= Ieee1394Camera.o \
		Ieee1394Node.o

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

include $(PROJECT)/lib/l.mk
###
Ieee1394Camera.o: Ieee1394++.h /Users/ueshiba/include/TU/Image++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Array++.cc \
	/Users/ueshiba/include/TU/Image++.cc \
	/Users/ueshiba/include/TU/functions.h \
	/Users/ueshiba/include/TU/mmInstructions.h
Ieee1394Node.o: Ieee1394++.h /Users/ueshiba/include/TU/Image++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
