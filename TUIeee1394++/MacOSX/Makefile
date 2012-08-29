#
#  $Id: Makefile,v 1.7 2012-08-29 19:33:23 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/libraw1394
INCDIRS		= -I$(PREFIX)/include

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
EXTHDRS		=
HDRS		= csr.h \
		raw1394.h \
		raw1394_.h
SRCS		= raw1394.cc
OBJS		= raw1394.o

include $(PROJECT)/lib/l.mk
###
raw1394.o: raw1394_.h raw1394.h
