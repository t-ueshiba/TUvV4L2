#
#  $Id: Makefile,v 1.4 2002-07-25 18:45:08 ueshiba Exp $
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
EXTHDRS		= /Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Object++.cc \
		/Users/ueshiba/include/TU/Object++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/types.h \
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
REV		= $(shell echo $Revision: 1.4 $	|		\
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
Geometry.o: TU/Brep/Brep++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h
HalfEdge.o: TU/Brep/Brep++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h
Loop.o: TU/Brep/Brep++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h
Neighbor.o: TU/Brep/Brep++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h
PointB.o: TU/Brep/Brep++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h
TUBrep++.inst.o: TU/Brep/Brep++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h \
	/Users/ueshiba/include/TU/Object++.cc
TUBrep++.sa.o: TU/Brep/Brep++.h /Users/ueshiba/include/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h
