#
#  $Id: Makefile,v 1.16 2008-06-27 07:53:20 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I. -I$(HOME)/include -I/usr/local/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DHAVE_TUToolsPP #-DRAW_ISO
CFLAGS		= -g
ifeq ($(CCC), icpc)
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS	= -O3 -axP -parallel -ip
  else
    CPPFLAGS   += -DSSE2
    CFLAGS	= -O3 -tpp7 -xW -ip
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Image++.cc \
		/Users/ueshiba/include/TU/Image++.h \
		/Users/ueshiba/include/TU/Minimize++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/mmInstructions.h \
		/Users/ueshiba/include/TU/types.h \
		/Users/ueshiba/include/TU/utility.h
HDRS		= Ieee1394++.h
SRCS		= Ieee1394Camera.cc \
		Ieee1394Node.cc
OBJS		= Ieee1394Camera.o \
		Ieee1394Node.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.16 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Ieee1394Camera.o: Ieee1394++.h /Users/ueshiba/include/TU/Image++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/utility.h \
	/Users/ueshiba/include/TU/Minimize++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Image++.cc \
	/Users/ueshiba/include/TU/mmInstructions.h
Ieee1394Node.o: Ieee1394++.h /Users/ueshiba/include/TU/Image++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/utility.h \
	/Users/ueshiba/include/TU/Minimize++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
