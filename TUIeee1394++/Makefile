#
#  $Id: Makefile,v 1.21 2008-09-02 05:15:32 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I. -I$(HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DHAVE_TUToolsPP
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
EXTHDRS		=
HDRS		= Ieee1394++.h
SRCS		= Ieee1394Camera.cc \
		Ieee1394Node.cc
OBJS		= Ieee1394Camera.o \
		Ieee1394Node.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.21 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Ieee1394Camera.o: Ieee1394++.h
Ieee1394Node.o: Ieee1394++.h
