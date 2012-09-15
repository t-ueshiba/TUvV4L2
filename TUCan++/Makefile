#
#  $Id: Makefile,v 1.16 2012-09-15 07:20:53 ueshiba Exp $
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
HDRS		= TU/Can++.h
SRCS		= Can.cc \
		Manus.cc
OBJS		= Can.o \
		Manus.o

include $(PROJECT)/lib/l.mk
###
Can.o: TU/Can++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
Manus.o: TU/Can++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
