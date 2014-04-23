#
#  $Id: Makefile,v 1.7 2012-09-15 07:21:10 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I. -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DNDEBUG -DHAVE_LIBTUTOOLS__
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
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/types.h
HDRS		= TU/V4L2++.h
SRCS		= V4L2Camera.cc
OBJS		= V4L2Camera.o

include $(PROJECT)/lib/l.mk
###
V4L2Camera.o: TU/V4L2++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h
