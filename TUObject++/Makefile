#
#  $Id: Makefile,v 1.3 2002-07-25 18:34:04 ueshiba Exp $
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
CCFLAGS		= -g

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/types.h \
		TU/Object++_.h \
		TU/TU/Object++.h
HDRS		= Object++.h \
		Object++_.h
SRCS		= Desc.cc \
		Object++.cc \
		Object.cc \
		Page.cc \
		TUObject+.sa.cc
OBJS		= Desc.o \
		Object++.o \
		Object.o \
		Page.o \
		TUObject+.sa.o

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

include $(PROJECT)/lib/l.mk
include $(PROJECT)/lib/RCS.mk
###
Desc.o: TU/Object++_.h TU/TU/Object++.h /Users/ueshiba/include/TU/types.h
Object++.o: TU/TU/Object++.h /Users/ueshiba/include/TU/types.h
Object.o: TU/Object++_.h TU/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
Page.o: TU/Object++_.h TU/TU/Object++.h /Users/ueshiba/include/TU/types.h
TUObject+.sa.o: TU/Object++_.h TU/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
