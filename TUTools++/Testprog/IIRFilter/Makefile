#
#  $Id: Makefile,v 1.9 2007-02-28 00:16:06 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I. -I$(HOME)/include

PROGRAM		= EdgeDetector
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
EXTHDRS		= /Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Image++.cc \
		/Users/ueshiba/include/TU/Image++.h \
		/Users/ueshiba/include/TU/Minimize++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/mmInstructions.h \
		/Users/ueshiba/include/TU/types.h \
		/Users/ueshiba/include/TU/utility.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

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
main.o: /Users/ueshiba/include/TU/Image++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/utility.h \
	/Users/ueshiba/include/TU/Minimize++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Image++.cc \
	/Users/ueshiba/include/TU/mmInstructions.h
