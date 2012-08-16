#
#  $Id: Makefile,v 1.12 2012-08-16 01:31:07 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= pstreetest
LIBS		= -lTUTools++

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
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Heap.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/PSTree.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= pstreetest.cc
OBJS		= pstreetest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.12 $	|		\
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
pstreetest.o: /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/PSTree.h /usr/local/include/TU/Heap.h \
	/usr/local/include/TU/List.h
