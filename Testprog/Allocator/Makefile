#
#  $Id: Makefile,v 1.7 2009-07-09 04:19:17 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= allocatortest
LIBS		= -lTUTools++

CPPFLAGS	= -DDEBUG
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		=

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /usr/local/include/TU/Allocator.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.7 $	|		\
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
main.o: /usr/local/include/TU/Allocator.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/List.h
