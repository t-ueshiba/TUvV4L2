#
#  $Id: Makefile,v 1.24 2012-09-15 05:16:29 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I. -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
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
EXTHDRS		=
HDRS		= TU/Collection++.h
SRCS		= Collection++.cc \
		ObjTreeBase.cc \
		TUCollection++.sa.cc
OBJS		= Collection++.o \
		ObjTreeBase.o \
		TUCollection++.sa.o

include $(PROJECT)/lib/l.mk
###
Collection++.o: TU/Collection++.h
ObjTreeBase.o: TU/Collection++.h
TUCollection++.sa.o: TU/Collection++.h
