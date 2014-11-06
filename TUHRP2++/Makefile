#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
#PROGRAM		= $(shell basename $(PWD))
LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		= ForwardKinematicsService.idl ReachingService.idl \
		  SequencePlayerService.idl WalkGeneratorService.idl
MOCHDRS		=

INCDIRS		= -I. -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifneq ($(findstring icpc,$(CXX)),)
  CFLAGS	= -O3
  NVCCFLAGS	= -O			# must < -O2
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		=
ifneq ($(findstring darwin,$(OSTYPE)),)
  LIBS	       += -framework IOKit -framework CoreFoundation \
		  -framework CoreServices
endif

LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= /usr/include/math.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/bodyinfo_HRP2DOF7.h \
		/usr/local/include/bodyinfo_HRP2YH.h \
		/usr/local/include/bodyinfo_HRP2toHRP3.h \
		/usr/local/include/bodyinfo_HRP3.h \
		/usr/local/include/misc.h
HDRS		= TU/HRP2++.h \
		bodyinfo_HRP2SH.h
SRCS		= HRP2.cc
OBJS		= HRP2.o

include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
HRP2.o: TU/HRP2++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/bodyinfo_HRP3.h /usr/local/include/misc.h \
	/usr/include/math.h /usr/local/include/bodyinfo_HRP2toHRP3.h \
	/usr/local/include/bodyinfo_HRP2YH.h bodyinfo_HRP2SH.h \
	/usr/local/include/bodyinfo_HRP2DOF7.h
