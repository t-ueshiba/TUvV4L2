#
#  $Id: Makefile,v 1.28 2012-08-15 07:58:34 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I../../

PROGRAM		= iirfprof
LIBS		= -lTUTools++ -ltbb
ifeq ($(OSTYPE), darwin)
  LIBS	       +=
#  LIBS	       += -framework IOKit -framework CoreFoundation
endif

CPPFLAGS	= -DUSE_TBB
CFLAGS		= -g -Wextra
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
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
EXTHDRS		= ../../TU/DericheConvolver.h \
		../../TU/GaussianConvolver.h \
		../../TU/Image++.h \
		../../TU/Profiler.h \
		../../TU/TU/Geometry++.h \
		../../TU/TU/IIRFilter.h \
		../../TU/TU/TU/Minimize.h \
		../../TU/TU/TU/TU/Array++.h \
		../../TU/TU/TU/TU/TU/types.h \
		../../TU/TU/TU/Vector++.h \
		../../TU/TU/TU/mmInstructions.h \
		../../TU/TU/TU/utility.h \
		../../TU/windows/fakeWindows.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.28 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
main.o: ../../TU/Image++.h ../../TU/TU/Geometry++.h \
	../../TU/TU/TU/utility.h ../../TU/TU/TU/Vector++.h \
	../../TU/TU/TU/TU/Array++.h ../../TU/TU/TU/TU/TU/types.h \
	../../TU/TU/TU/Minimize.h ../../TU/DericheConvolver.h \
	../../TU/TU/IIRFilter.h ../../TU/TU/TU/mmInstructions.h \
	../../TU/GaussianConvolver.h ../../TU/Profiler.h \
	../../TU/windows/fakeWindows.h
