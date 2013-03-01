#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I$(PREFIX)/include

PROGRAM		= HRP2test
LIBS		= -lTUHRP2++ -lTUTools++ -lHRP2Corba `rtm-config --libs`

ifeq ($(OSTYPE), darwin)
  LIBS	       +=
endif

#CXX		= g++
CPPFLAGS	=
CFLAGS		= -g `rtm-config --cflags`
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS     += -xSSE3
  else
    CPPFLAGS   += -DSSE3
    CFLAGS     += -xSSE3
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= /usr/include/math.h \
		/usr/local/include/ForwardKinematicsService.hh \
		/usr/local/include/ReachingService.h \
		/usr/local/include/SequencePlayerService.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/HRP2++.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/WalkGeneratorService.h \
		/usr/local/include/bodyinfo_HRP2DOF7.h \
		/usr/local/include/bodyinfo_HRP2YH.h \
		/usr/local/include/bodyinfo_HRP2toHRP3.h \
		/usr/local/include/bodyinfo_HRP3.h \
		/usr/local/include/misc.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision$	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
main.o: /usr/local/include/TU/Manip.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/HRP2++.h /usr/local/include/ReachingService.h \
	/usr/local/include/SequencePlayerService.h \
	/usr/local/include/ForwardKinematicsService.hh \
	/usr/local/include/WalkGeneratorService.h /usr/local/include/misc.h \
	/usr/include/math.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/bodyinfo_HRP3.h \
	/usr/local/include/bodyinfo_HRP2toHRP3.h \
	/usr/local/include/bodyinfo_HRP2YH.h \
	/usr/local/include/bodyinfo_HRP2DOF7.h
