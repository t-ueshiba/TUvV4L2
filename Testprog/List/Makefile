#
#  $Id: Makefile,v 1.4 2008-09-02 05:13:16 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include

PROGRAM		= slisttest
LIBS		=

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/List.h
HDRS		=
SRCS		= slisttest.cc
OBJS		= slisttest.o

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
slisttest.o: /home/ueshiba/include/TU/List.h
