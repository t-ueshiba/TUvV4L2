#
#  $Id: Makefile,v 1.2 2012-04-20 00:53:14 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include -I$(PORTSHOME)/include

PROGRAM		= gctest
LIBS		=

CPPFLAGS	= -DWITH_PARALLEL_EDGES #-D_DEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
  else
    CPPFLAGS   += -DSSSE3
    CFLAGS     += -xSSSE3
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= /usr/local/include/TU/GraphCuts.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.2 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
main.o: /usr/local/include/TU/GraphCuts.h
