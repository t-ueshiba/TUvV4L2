#
#  $Id: Makefile,v 1.9 2008-09-08 08:15:55 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I. -I$(HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3 -axP -ip -parallel
  else
    CPPFLAGS   += -DSSE2 -xN -ip
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Camera.h \
		/home/ueshiba/include/TU/Geometry++.h \
		/home/ueshiba/include/TU/Image++.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/Warp.h \
		/home/ueshiba/include/TU/types.h \
		TU/Thread++.h
HDRS		= IIRFilterMT.h \
		Thread++.h
SRCS		= Thread.cc \
		WarpMT.cc
OBJS		= Thread.o \
		WarpMT.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.9 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Thread.o: TU/Thread++.h /home/ueshiba/include/TU/Array++.h \
	/home/ueshiba/include/TU/types.h
WarpMT.o: TU/Thread++.h /home/ueshiba/include/TU/Array++.h \
	/home/ueshiba/include/TU/types.h /home/ueshiba/include/TU/Warp.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h /home/ueshiba/include/TU/Camera.h
