#
#  $Id: Makefile,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
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
EXTHDRS		= /home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Image++.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/types.h \
		TU/S2200++.h
HDRS		= S2200++.h
SRCS		= S2200.cc
OBJS		= S2200.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.1.1.1 $	|		\
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
S2200.o: TU/S2200++.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
