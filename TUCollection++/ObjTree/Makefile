#
#  $Id: Makefile,v 1.2 2005-05-10 02:51:14 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)
INCDIRS		= -I../ObjList -I$(HOME)/include

PROGRAM		= treetest
LIBS		= -lTUCollection++ -lTUObject++

CPPFLAGS	= -DTUCollectionPP_DEBUG
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)
LINKER		= $(CCC)
ifeq ($(CCC), icc)
#  CCFLAGS	= -cxxlib-icc -no-gcc -O3 -parallel
  CCFLAGS	= -cxxlib-icc -no-gcc -g
  LDFLAGS      += -cxxlib-icc
  LINKER	= icpc
endif

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Collection++.h \
		/home/ueshiba/include/TU/Object++.h \
		/home/ueshiba/include/TU/types.h
HDRS		=
SRCS		= Int.cc \
		print.cc \
		treetest.cc
OBJS		= Int.o \
		print.o \
		treetest.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.2 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk
###
print.o: /home/ueshiba/include/TU/Collection++.h \
	/home/ueshiba/include/TU/Object++.h /home/ueshiba/include/TU/types.h
treetest.o: /home/ueshiba/include/TU/Collection++.h \
	/home/ueshiba/include/TU/Object++.h /home/ueshiba/include/TU/types.h
