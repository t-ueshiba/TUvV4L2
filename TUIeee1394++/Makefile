#
#  $Id: Makefile,v 1.30 2009-05-10 23:34:27 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I. -I$(HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DHAVE_LIBTUTOOLS__ # -DUSE_RAWISO
CFLAGS		= -g
ifeq ($(CCC), icpc)
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS	= -O3 -axP
  else
    CPPFLAGS   += -DSSSE3
    CFLAGS	= -O3 -xN
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Geometry++.h \
		/home/ueshiba/include/TU/Image++.h \
		/home/ueshiba/include/TU/Normalize.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/types.h \
		TU/Ieee1394++.h \
		TU/Ieee1394CameraArray.h
HDRS		= Ieee1394++.h \
		Ieee1394CameraArray.h
SRCS		= Ieee1394Camera.cc \
		Ieee1394CameraArray.cc \
		Ieee1394Node.cc
OBJS		= Ieee1394Camera.o \
		Ieee1394CameraArray.o \
		Ieee1394Node.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.30 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Ieee1394Camera.o: TU/Ieee1394++.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
Ieee1394CameraArray.o: TU/Ieee1394CameraArray.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	TU/Ieee1394++.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Normalize.h
Ieee1394Node.o: TU/Ieee1394++.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
