#
#  $Id: Makefile,v 1.7 2005-05-10 02:51:14 ueshiba Exp $
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
ifeq ($(CCC), icc)
#  CCFLAGS	= -cxxlib-icc -O3 -parallel
  CCFLAGS	= -cxxlib-icc -g
  LDFLAGS      += -cxxlib-icc
  LINKER	= icpc
endif

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Object++.cc \
		/home/ueshiba/include/TU/Object++.h \
		/home/ueshiba/include/TU/types.h
HDRS		=
SRCS		= ptest.cc
OBJS		= ptest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.7 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
###
ptest.o: /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h /home/ueshiba/include/TU/Object++.cc
