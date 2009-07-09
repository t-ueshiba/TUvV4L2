#
#  $Id: Makefile,v 1.9 2009-07-09 04:19:26 ueshiba Exp $
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
EXTHDRS		= /home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Geometry++.h \
		/home/ueshiba/include/TU/Heap.h \
		/home/ueshiba/include/TU/List.h \
		/home/ueshiba/include/TU/Normalize.h \
		/home/ueshiba/include/TU/PSTree.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/types.h
HDRS		=
SRCS		= pstreetest.cc
OBJS		= pstreetest.o

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
include $(PROJECT)/lib/RCS.mk
###
pstreetest.o: /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h \
	/home/ueshiba/include/TU/PSTree.h /home/ueshiba/include/TU/Heap.h \
	/home/ueshiba/include/TU/List.h
