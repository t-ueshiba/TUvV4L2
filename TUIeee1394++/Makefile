#
#  $Id: Makefile,v 1.8 2003-02-20 05:51:15 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include -I/usr/local/include
SRCDIR		= $(HOME)/VVV/Pkg/vvv/libraries/libTUIeee1394++

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DHAVE_TUToolsPP
CFLAGS		= -O -g
ifeq ($(CCC), icc)
#  CCFLAGS	= -O3 -tpp7 -xW -parallel
  CCFLAGS	= -O
else
  CCFLAGS	= -O -g
endif

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Array++.cc \
		/home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Image++.cc \
		/home/ueshiba/include/TU/Image++.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/types.h \
		TU/Ieee1394++.h
HDRS		= Ieee1394++.h
SRCS		= Ieee1394Camera.cc \
		Ieee1394Node.cc \
		Ieee1394Port.cc
OBJS		= Ieee1394Camera.o \
		Ieee1394Node.o \
		Ieee1394Port.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.8 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
include $(PROJECT)/lib/l-install-srcs.mk
###
Ieee1394Camera.o: TU/Ieee1394++.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Array++.cc \
	/home/ueshiba/include/TU/Image++.cc
Ieee1394Node.o: TU/Ieee1394++.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Ieee1394Port.o: TU/Ieee1394++.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
