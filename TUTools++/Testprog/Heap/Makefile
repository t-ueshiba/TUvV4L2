#
#  $Id: Makefile,v 1.2 2002-07-26 08:59:31 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include

PROGRAM		= heaptest
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
		/Users/ueshiba/include/TU/Heap++.cc \
		/Users/ueshiba/include/TU/Heap++.h \
		/Users/ueshiba/include/TU/types.h
HDRS		=
SRCS		= heaptest.cc
OBJS		= heaptest.o

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
heaptest.o: /Users/ueshiba/include/TU/Heap++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Heap++.cc
