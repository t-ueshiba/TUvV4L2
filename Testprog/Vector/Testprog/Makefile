#
#  $Id: Makefile,v 1.4 2002-08-12 02:38:23 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)			# destination of the program
INCDIRS		= -I$(HOME)/include

PROGRAM		= mtest
LIBS		= -lTUTools++

CPPFLAGS	= -g
CFLAGS		= -g
CCFLAGS		= 
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
HDRS		=
SRCS		= mtest.cc
OBJS		= mtest.o

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
mtest.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
