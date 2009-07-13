#
#  $Id: Makefile,v 1.24 2009-07-13 01:15:07 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/TU/Brep
INCDIRS		= -I. -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DTUBrepPP_DEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS     += -xSSE3
  else
    CPPFLAGS   += -DSSE3
    CFLAGS     += -xSSE3
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Normalize.h \
		/usr/local/include/TU/Object++.cc \
		/usr/local/include/TU/Object++.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/types.h \
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
REV		= $(shell echo $Revision: 1.24 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Geometry.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Normalize.h
HalfEdge.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Normalize.h
Loop.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Normalize.h
Neighbor.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Normalize.h
PointB.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Normalize.h
TUBrep++.inst.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Normalize.h /usr/local/include/TU/Object++.cc
TUBrep++.sa.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Normalize.h
