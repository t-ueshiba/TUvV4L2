#
#  $Id: Makefile,v 1.1 2002-07-25 04:35:50 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)			# destination of the program
INCDIRS		= -I../ObjList -I$(HOME)/include

PROGRAM		= dlisttest
LIBS		= -lTUCollection++ -lTUObject++ -lTUTools++

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= 
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= ../ObjList/Int.h \
		/home/ueshiba/include/TU/Collection++.h \
		/home/ueshiba/include/TU/Object++.h \
		/home/ueshiba/include/TU/types.h
HDRS		=
SRCS		= Int.cc \
		dlisttest.cc
OBJS		= Int.o \
		dlisttest.o

#########################
#  Macros used by RCS	#
#########################
REV		= `echo $Revision: 1.1 $	|			\
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
Int.o: ../ObjList/Int.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h
dlisttest.o: ../ObjList/Int.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Collection++.h
