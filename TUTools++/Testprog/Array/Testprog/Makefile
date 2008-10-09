#
#  $Id: Makefile,v 1.4 2008-10-09 05:52:14 ueshiba Exp $
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
EXTHDRS		= /home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/types.h
HDRS		=
SRCS		= test.cc
OBJS		= test.o

#########################
#  Macros used by RCS	#
#########################
REV		= `echo $Revision: 1.4 $	|			\
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
test.o: /home/ueshiba/include/TU/Array++.h \
	/home/ueshiba/include/TU/types.h
