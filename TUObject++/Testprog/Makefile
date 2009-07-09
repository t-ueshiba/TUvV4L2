#
#  $Id: Makefile,v 1.9 2009-07-09 04:58:26 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= ptest
LIBS		= -lTUObject++

CPPFLAGS	= -DTUObjectPP_DEBUG
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		=
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /usr/local/include/TU/Object++.cc \
		/usr/local/include/TU/Object++.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= ptest.cc
OBJS		= ptest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.9 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
###
ptest.o: /usr/local/include/TU/Object++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Object++.cc
