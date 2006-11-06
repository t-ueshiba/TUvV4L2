#
#  $Id: Makefile,v 1.3 2006-11-06 02:17:41 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include

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
EXTHDRS		= /Users/ueshiba/include/TU/Array++.cc \
		/Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Image++.cc \
		/Users/ueshiba/include/TU/Image++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/mmInstructions.h \
		/Users/ueshiba/include/TU/types.h
HDRS		= EdgeDetector.h
SRCS		= EdgeDetector.cc \
		main.cc
OBJS		= EdgeDetector.o \
		main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.3 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
EdgeDetector.o: EdgeDetector.h /Users/ueshiba/include/TU/Image++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/mmInstructions.h
main.o: /Users/ueshiba/include/TU/Image++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	EdgeDetector.h /Users/ueshiba/include/TU/Array++.cc \
	/Users/ueshiba/include/TU/Image++.cc \
	/Users/ueshiba/include/TU/mmInstructions.h
