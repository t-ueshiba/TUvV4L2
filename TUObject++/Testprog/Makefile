#
#  $Id: Makefile,v 1.2 2002-07-25 07:21:06 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include

PROGRAM		= ptest
LIBS		= -lTUObject++

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home1/ueshiba/include/TU/Object++.cc \
		/home1/ueshiba/include/TU/Object++.h \
		/home1/ueshiba/include/TU/types.h
HDRS		=
SRCS		= ptest.cc
OBJS		= ptest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.2 $	|		\
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
ptest.o: /home1/ueshiba/include/TU/Object++.h \
	/home1/ueshiba/include/TU/types.h \
	/home1/ueshiba/include/TU/Object++.cc
