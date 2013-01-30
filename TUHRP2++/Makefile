#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I. -I$(PREFIX)/include -I$(PREFIX)/include/rtm/idl

NAME		= $(shell basename $(PWD))

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
EXTHDRS		= /usr/include/math.h \
		/usr/local/include/ForwardKinematicsService.hh \
		/usr/local/include/HRP2Client.h \
		/usr/local/include/ReachingService.h \
		/usr/local/include/SequencePlayerService.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/WalkGeneratorService.h \
		/usr/local/include/bodyinfo_HRP2DOF7.h \
		/usr/local/include/bodyinfo_HRP2YH.h \
		/usr/local/include/bodyinfo_HRP2toHRP3.h \
		/usr/local/include/bodyinfo_HRP3.h \
		/usr/local/include/misc.h
HDRS		= TU/HRP2++.h
SRCS		= HRP2.cc
OBJS		= HRP2.o

include $(PROJECT)/lib/l.mk
###
HRP2.o: TU/HRP2++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/HRP2Client.h /usr/local/include/ReachingService.h \
	/usr/local/include/SequencePlayerService.h \
	/usr/local/include/ForwardKinematicsService.hh \
	/usr/local/include/WalkGeneratorService.h /usr/local/include/misc.h \
	/usr/include/math.h /usr/local/include/bodyinfo_HRP3.h \
	/usr/local/include/bodyinfo_HRP2toHRP3.h \
	/usr/local/include/bodyinfo_HRP2YH.h \
	/usr/local/include/bodyinfo_HRP2DOF7.h
