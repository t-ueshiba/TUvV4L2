#
#  $Id: Makefile,v 1.5 2005-05-10 05:20:24 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I$(HOME)/include

PROGRAM		= listtest
LIBS		= -lTUCollection++ -lTUObject++

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Collection++.cc \
		/home/ueshiba/include/TU/Collection++.h \
		/home/ueshiba/include/TU/Object++.h \
		/home/ueshiba/include/TU/types.h
HDRS		= Int.h
SRCS		= Int.cc \
		listtest.cc
OBJS		= Int.o \
		listtest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.5 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
###
Int.o: Int.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h
listtest.o: Int.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Collection++.h \
	/home/ueshiba/include/TU/Collection++.cc
