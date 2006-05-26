#
#  $Id: Makefile,v 1.13 2006-05-26 07:30:19 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include

PROGRAM		= mtest
LIBS		= -lTUTools++

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/types.h
HDRS		=
SRCS		= mtest.cc
OBJS		= mtest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.13 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
###
mtest.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
