#
#  $Id: Makefile,v 1.11 2007-02-28 00:18:40 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DTUObjectPP_DEBUG
CFLAGS		= -O -g
CCFLAGS		= -O -g
ifeq ($(CCC), icpc)
  CCFLAGS	= -O3 -parallel
endif
LDFLAGS		= $(CCFLAGS)
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/types.h \
		TU/Object++.h
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
REV		= $(shell echo $Revision: 1.11 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Desc.o: Object++_.h TU/Object++.h /Users/ueshiba/include/TU/types.h
Object++.o: TU/Object++.h /Users/ueshiba/include/TU/types.h
Object.o: Object++_.h TU/Object++.h /Users/ueshiba/include/TU/types.h
Page.o: Object++_.h TU/Object++.h /Users/ueshiba/include/TU/types.h
TUObject++.sa.o: Object++_.h TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
