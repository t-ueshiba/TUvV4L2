#
#  $Id: Makefile,v 1.4 2002-07-26 08:56:04 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I.. -I$(HOME)/include

PROGRAM		= ptest
LIBS		= -lTUObject++

CPPFLAGS	= -DTUObjectPP_DEBUG
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= ../Object++_.h \
		../TU/Object++.cc \
		../TU/Object++.h \
		/Users/ueshiba/include/TU/types.h
HDRS		=
SRCS		= Desc.cc \
		ptest.cc
OBJS		= Desc.o \
		ptest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.4 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
include $(PROJECT)/lib/RCS.mk
###
Desc.o: ../Object++_.h ../TU/Object++.h /Users/ueshiba/include/TU/types.h
ptest.o: ../TU/Object++.h /Users/ueshiba/include/TU/types.h \
	../TU/Object++.cc
