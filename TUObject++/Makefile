#
#  $Id: Makefile,v 1.16 2008-09-02 05:15:32 ueshiba Exp $
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
EXTHDRS		= TU/Object++.h
HDRS		= Object++.h \
		Object++_.h
SRCS		= Desc.cc \
		Object++.cc \
		Object.cc \
		Page.cc \
		TUObject++.sa.cc
OBJS		= Desc.o \
		Object++.o \
		Object.o \
		Page.o \
		TUObject++.sa.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.16 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Desc.o: Object++_.h TU/Object++.h
Object++.o: TU/Object++.h
Object.o: Object++_.h TU/Object++.h
Page.o: Object++_.h TU/Object++.h
TUObject++.sa.o: Object++_.h TU/Object++.h
