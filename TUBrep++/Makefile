#
#  $Id: Makefile,v 1.3 2002-07-25 07:57:45 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU/Brep
INCDIRS		= -I$(HOME)/include

NAME		= TUBrep++

CPPFLAGS	= -DTUBrepPP_DEBUG
CFLAGS		= -O
CCFLAGS		= -O

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home1/ueshiba/include/TU/Object++.cc \
		/home1/ueshiba/include/TU/Object++.h \
		/home1/ueshiba/include/TU/types.h \
		TU/Brep/Brep++.h
HDRS		= Brep++.h
SRCS		= Geometry.cc \
		HalfEdge.cc \
		Loop.cc \
		Neighbor.cc \
		PointB.cc \
		TUBrep++.inst.cc \
		TUBrep++.sa.cc
OBJS		= Geometry.o \
		HalfEdge.o \
		Loop.o \
		Neighbor.o \
		PointB.o \
		TUBrep++.inst.o \
		TUBrep++.sa.o

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
Geometry.o: TU/Brep/Brep++.h /home1/ueshiba/include/TU/Object++.h \
	/home1/ueshiba/include/TU/types.h
HalfEdge.o: TU/Brep/Brep++.h /home1/ueshiba/include/TU/Object++.h \
	/home1/ueshiba/include/TU/types.h
Loop.o: TU/Brep/Brep++.h /home1/ueshiba/include/TU/Object++.h \
	/home1/ueshiba/include/TU/types.h
Neighbor.o: TU/Brep/Brep++.h /home1/ueshiba/include/TU/Object++.h \
	/home1/ueshiba/include/TU/types.h
PointB.o: TU/Brep/Brep++.h /home1/ueshiba/include/TU/Object++.h \
	/home1/ueshiba/include/TU/types.h
TUBrep++.inst.o: TU/Brep/Brep++.h /home1/ueshiba/include/TU/Object++.h \
	/home1/ueshiba/include/TU/types.h \
	/home1/ueshiba/include/TU/Object++.cc
TUBrep++.sa.o: TU/Brep/Brep++.h /home1/ueshiba/include/TU/Object++.h \
	/home1/ueshiba/include/TU/types.h
