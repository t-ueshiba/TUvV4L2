#
#  $Id: Makefile,v 1.3 2002-07-25 18:34:04 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include

PROGRAM		= ptest
LIBS		= -lTUObject++

CPPFLAGS	= -DTUObjectPP_DEBUG
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/types.h \
		TU/Object++.cc \
		TU/Object++_.h \
		TU/TU/Object++.h
HDRS		=
SRCS		= Desc.cc \
		TUObject+.sa.cc \
		ptest.cc
OBJS		= Desc.o \
		TUObject+.sa.o \
		ptest.o

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
include $(PROJECT)/lib/RCS.mk
###
Desc.o: TU/Object++_.h TU/TU/Object++.h /Users/ueshiba/include/TU/types.h
TUObject+.sa.o: TU/Object++_.h TU/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
ptest.o: TU/TU/Object++.h /Users/ueshiba/include/TU/types.h TU/Object++.cc
