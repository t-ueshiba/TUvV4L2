#
#  $Id: Makefile,v 1.1 2002-07-25 04:36:08 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include #-I$(HOME)/include/stl

PROGRAM		= allocatortest
LIBS		= -lTUTools++

CPPFLAGS	= -DDEBUG
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Allocator++.cc \
		/home/ueshiba/include/TU/Allocator++.h \
		/home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/List++.cc \
		/home/ueshiba/include/TU/List++.h \
		/home/ueshiba/include/TU/types.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.1 $	|		\
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
main.o: /home/ueshiba/include/TU/Allocator++.h \
	/home/ueshiba/include/TU/List++.h /home/ueshiba/include/TU/Array++.h \
	/home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Allocator++.cc \
	/home/ueshiba/include/TU/List++.cc
