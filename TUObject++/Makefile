#
#  $Id: Makefile,v 1.24 2012-09-01 05:37:10 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/TU
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
EXTHDRS		= /usr/local/include/TU/types.h \
		TU/Object++.h
HDRS		= Object++.h \
		Object++_.h
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
