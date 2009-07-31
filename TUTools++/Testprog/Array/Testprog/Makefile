#
#  $Id: Makefile,v 1.6 2009-07-31 07:05:03 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= test
LIBS		= -lTUTools++
ifeq ($(OSTYPE), darwin)
  LIBS	       +=
#  LIBS	       += -framework IOKit -framework CoreFoundation
endif

CPPFLAGS	=
CFLAGS		= -g -Wextra
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
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= test.cc
OBJS		= test.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.6 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
test.o: /usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
