#
#  $Id: Makefile,v 1.1 2002-07-25 04:36:08 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(BINDIR)			# destination of the program
INCDIRS		= -I../ObjList -I$(HOME)/include

PROGRAM		= treetest
LIBS		= -lTUCollection++ -lTUObject++ -lTUTools++

CPPFLAGS	= -DTUCollectionPP_DEBUG
CFLAGS		= -g
CCFLAGS		= 
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= ../ObjList/Int.h \
		/home/ueshiba/include/TU/Collection++.h \
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
REV		= `echo $Revision: 1.1 $	|			\
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
Int.o: ../ObjList/Int.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h
print.o: /home/ueshiba/include/TU/Collection++.h \
	/home/ueshiba/include/TU/Object++.h /home/ueshiba/include/TU/types.h \
	../ObjList/Int.h
treetest.o: ../ObjList/Int.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Collection++.h
