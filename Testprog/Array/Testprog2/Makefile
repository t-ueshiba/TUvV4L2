#
#  $Id: Makefile,v 1.2 2002-08-12 01:24:46 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)			# destination of the program
INCDIRS		= -I$(HOME)/include

PROGRAM		= test2
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
		/Users/ueshiba/include/TU/types.h
HDRS		=
SRCS		= test2.cc
OBJS		= test2.o

#########################
#  Macros used by RCS	#
#########################
REV		= `echo $Revision: 1.2 $	|			\
		  sed 's/evision://'	|			\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }'`

include $(PROJECT)/lib/p.mk
include $(PROJECT)/lib/RCS.mk
###
test2.o: /Users/ueshiba/include/TU/Array++.h \
	/Users/ueshiba/include/TU/types.h
