#
#  $Id: Makefile,v 1.5 2002-07-29 00:09:06 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include

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
EXTHDRS		= /Users/ueshiba/include/TU/Object++.cc \
		/Users/ueshiba/include/TU/Object++.h \
		/Users/ueshiba/include/TU/types.h
HDRS		=
SRCS		= ptest.cc
OBJS		= ptest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.5 $	|		\
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
ptest.o: /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Object++.cc
