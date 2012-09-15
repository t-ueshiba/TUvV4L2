#
#  $Id: Makefile,v 1.8 2012-09-15 05:13:16 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I. -I$(PREFIX)/include

NAME		= raw1394

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
EXTHDRS		= raw1394_.h
HDRS		= raw1394_.h
SRCS		= raw1394.cc
OBJS		= raw1394.o

include $(PROJECT)/lib/l.mk
###
raw1394.o: raw1394_.h libraw1394/raw1394.h
