#
#  $Id: Makefile,v 1.7 2012-08-15 07:58:34 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= heaptest
LIBS		= -lTUTools++

#CCC		= g++
CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		=

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Heap.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= heaptest.cc
OBJS		= heaptest.o

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
heaptest.o: /usr/local/include/TU/Heap.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
