#
#  $Id: Makefile,v 1.4 2012-07-28 09:10:24 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= 

PROGRAM		= iirftest
LIBS		=

CPPFLAGS	= -std=c++0x -DSSE2
CFLAGS		= -O3
CCFLAGS		= $(CFLAGS)
LDFLAGS		=

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= TU/DericheConvolver.h \
		TU/Image++.h \
		TU/TU/Geometry++.h \
		TU/TU/IIRFilter.h \
		TU/TU/TU/Minimize.h \
		TU/TU/TU/TU/Array++.h \
		TU/TU/TU/TU/TU/types.h \
		TU/TU/TU/Vector++.h \
		TU/TU/TU/mmInstructions.h \
		TU/TU/TU/utility.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.4 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
main.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/utility.h \
	TU/TU/TU/Vector++.h TU/TU/TU/TU/Array++.h TU/TU/TU/TU/TU/types.h \
	TU/TU/TU/Minimize.h TU/DericheConvolver.h TU/TU/IIRFilter.h \
	TU/TU/TU/mmInstructions.h
