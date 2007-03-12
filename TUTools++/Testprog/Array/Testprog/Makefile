#
#  $Id: Makefile,v 1.3 2007-03-12 07:15:40 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)			# destination of the program
INCDIRS		= -I$(HOME)/include

PROGRAM		= test
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
EXTHDRS		= TU/Array++.h \
		TU/TU/types.h
HDRS		=
SRCS		= test.cc
OBJS		= test.o

#########################
#  Macros used by RCS	#
#########################
REV		= `echo $Revision: 1.3 $	|			\
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
test.o: TU/Array++.h TU/TU/types.h
