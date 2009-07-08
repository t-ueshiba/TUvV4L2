#
#  $Id: Makefile,v 1.20 2009-07-08 01:10:18 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/TU
INCDIRS		= -I. -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
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
EXTHDRS		= /usr/local/include/TU/Object++.h \
		/usr/local/include/TU/types.h \
		TU/Collection++.h
HDRS		= Collection++.h
SRCS		= Collection++.cc \
		ObjTreeBase.cc \
		TUCollection++.sa.cc
OBJS		= Collection++.o \
		ObjTreeBase.o \
		TUCollection++.sa.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.20 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Collection++.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
ObjTreeBase.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
TUCollection++.sa.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
