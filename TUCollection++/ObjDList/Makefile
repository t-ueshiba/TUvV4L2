#
#  $Id: Makefile,v 1.2 2002-07-26 08:57:37 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)			# destination of the program
INCDIRS		= -I../ObjList -I../../TUObject++ -I$(HOME)/include

PROGRAM		= dlisttest
LIBS		= -lTUCollection++ -lTUObject++ -lTUTools++

CPPFLAGS	= -DTUObjectPP_DEBUG
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= ../../TUObject++/Object++_.h \
		../../TUObject++/TU/Object++.h \
		../ObjList/Int.h \
		/Users/ueshiba/include/TU/Collection++.cc \
		/Users/ueshiba/include/TU/Collection++.h \
		/Users/ueshiba/include/TU/types.h
HDRS		=
SRCS		= Desc.cc \
		Int.cc \
		dlisttest.cc
OBJS		= Desc.o \
		Int.o \
		dlisttest.o

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
Desc.o: ../../TUObject++/Object++_.h ../../TUObject++/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
Int.o: ../ObjList/Int.h ../../TUObject++/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
dlisttest.o: ../ObjList/Int.h ../../TUObject++/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Collection++.h \
	/Users/ueshiba/include/TU/Collection++.cc
