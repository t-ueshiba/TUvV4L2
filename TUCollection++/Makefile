#
#  $Id: Makefile,v 1.22 2012-08-29 21:16:57 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/TU
INCDIRS		= -I$(PREFIX)/include

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
EXTHDRS		= /usr/local/include/TU/Object++.h \
		/usr/local/include/TU/types.h \
		TU/Collection++.h
HDRS		= Collection++.h
SRCS		= Collection++.cc \
		ObjTreeBase.cc \
		TUCollection++.sa.cc
OBJS		= Collection++.o \
		ObjTreeBase.o \
		TUCollection++.sa.o

include $(PROJECT)/lib/l.mk
###
Collection++.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
ObjTreeBase.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
TUCollection++.sa.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
