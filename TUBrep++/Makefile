#
#  $Id: Makefile,v 1.26 2012-08-29 21:16:44 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/TU/Brep
INCDIRS		= -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DTUBrepPP_DEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Object++.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
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

include $(PROJECT)/lib/l.mk
###
Geometry.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Minimize.h
HalfEdge.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Minimize.h
Loop.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Minimize.h
Neighbor.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Minimize.h
PointB.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Minimize.h
TUBrep++.inst.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Minimize.h
TUBrep++.sa.o: TU/Brep/Brep++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/Minimize.h
