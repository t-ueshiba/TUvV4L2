#
#  $Id: Makefile,v 1.3 2002-07-25 18:45:57 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -O
CCFLAGS		= -O

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Object++.h \
		/Users/ueshiba/include/TU/types.h \
		TU/Collection++.h
HDRS		= Collection++.h
SRCS		= Collection++.cc \
		ObjTreeBase.cc \
		TUCollection++.sa.cc
OBJS		= Collection++.o \
		ObjTreeBase.o \
		TUCollection++.sa.o

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

include $(PROJECT)/lib/l.mk
include $(PROJECT)/lib/RCS.mk
###
Collection++.o: TU/Collection++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
ObjTreeBase.o: TU/Collection++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
TUCollection++.sa.o: TU/Collection++.h \
	/Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
