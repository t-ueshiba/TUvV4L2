#
#  $Id: Makefile,v 1.10 2005-05-10 05:20:24 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DTUCollectionPP_DEBUG
CFLAGS		= -O -g
CCFLAGS		= -O -g
ifeq ($(CCC), icpc)
  CCFLAGS	= -O3 -parallel
endif
LDFLAGS		= $(CCFLAGS)
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Object++.h \
		/home/ueshiba/include/TU/types.h \
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
REV		= $(shell echo $Revision: 1.10 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Collection++.o: TU/Collection++.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h
ObjTreeBase.o: TU/Collection++.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h
TUCollection++.sa.o: TU/Collection++.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h
