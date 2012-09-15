#
#  $Id: Makefile,v 1.26 2012-09-15 07:21:03 ueshiba Exp $
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
EXTHDRS		= /usr/local/include/TU/types.h
HDRS		= Object++_.h \
		TU/Object++.h
SRCS		= Desc.cc \
		Object++.cc \
		Object.cc \
		Page.cc \
		TUObject++.sa.cc
OBJS		= Desc.o \
		Object++.o \
		Object.o \
		Page.o \
		TUObject++.sa.o

include $(PROJECT)/lib/l.mk
###
Desc.o: Object++_.h TU/Object++.h /usr/local/include/TU/types.h
Object++.o: TU/Object++.h /usr/local/include/TU/types.h
Object.o: Object++_.h TU/Object++.h /usr/local/include/TU/types.h
Page.o: Object++_.h TU/Object++.h /usr/local/include/TU/types.h
TUObject++.sa.o: Object++_.h TU/Object++.h /usr/local/include/TU/types.h
