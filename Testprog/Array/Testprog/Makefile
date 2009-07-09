#
#  $Id: Makefile,v 1.5 2009-07-09 04:19:26 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin			# destination of the program
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= test
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
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= test.cc
OBJS		= test.o

#########################
#  Macros used by RCS	#
#########################
REV		= `echo $Revision: 1.5 $	|			\
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
test.o: /usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
