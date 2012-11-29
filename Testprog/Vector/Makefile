#
#  $Id: Makefile,v 1.10 2012-08-31 02:37:12 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= mtest
LIBS		=
ifeq ($(OSTYPE), darwin)
  LIBS	       +=
#  LIBS	       += -framework IOKit -framework CoreFoundation -framework CoreServices
endif

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
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Expression++.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= mtest.cc
OBJS		= mtest.o

include $(PROJECT)/lib/p.mk

###
mtest.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Expression++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
