#
#  $Id: Makefile,v 1.6 2009-07-09 04:58:26 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= listtest
LIBS		= -lTUCollection++ -lTUObject++

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		=
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /usr/local/include/TU/Collection++.cc \
		/usr/local/include/TU/Collection++.h \
		/usr/local/include/TU/Object++.h \
		/usr/local/include/TU/types.h
HDRS		= Int.h
SRCS		= Int.cc \
		listtest.cc
OBJS		= Int.o \
		listtest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.6 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
###
Int.o: Int.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
listtest.o: Int.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Collection++.h \
	/usr/local/include/TU/Collection++.cc
