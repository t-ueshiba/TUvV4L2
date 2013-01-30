#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I.. -I$(PREFIX)/include -I$(PREFIX)/include/rtm/idl

PROGRAM		= HRP2test
LIBS		= -lTUTools++ -lHRP2Client -lHRP2Corba `rtm-config --libs`
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
EXTHDRS		= ../TU/HRP2++.h \
		/usr/include/math.h \
		/usr/local/include/ForwardKinematicsService.hh \
		/usr/local/include/HRP2Client.h \
		/usr/local/include/ReachingService.h \
		/usr/local/include/SequencePlayerService.h \
		/usr/local/include/TU/Array++.h \
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
SRCS		= HRP2.cc \
		main.cc
OBJS		= HRP2.o \
		main.o

include $(PROJECT)/lib/p.mk

###
HRP2.o: ../TU/HRP2++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/HRP2Client.h /usr/local/include/ReachingService.h \
	/usr/local/include/SequencePlayerService.h \
	/usr/local/include/ForwardKinematicsService.hh \
	/usr/local/include/WalkGeneratorService.h /usr/local/include/misc.h \
	/usr/include/math.h /usr/local/include/bodyinfo_HRP3.h \
	/usr/local/include/bodyinfo_HRP2toHRP3.h \
	/usr/local/include/bodyinfo_HRP2YH.h \
	/usr/local/include/bodyinfo_HRP2DOF7.h
main.o: /usr/local/include/TU/Manip.h /usr/local/include/TU/types.h \
	../TU/HRP2++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/HRP2Client.h \
	/usr/local/include/ReachingService.h \
	/usr/local/include/SequencePlayerService.h \
	/usr/local/include/ForwardKinematicsService.hh \
	/usr/local/include/WalkGeneratorService.h /usr/local/include/misc.h \
	/usr/include/math.h /usr/local/include/bodyinfo_HRP3.h \
	/usr/local/include/bodyinfo_HRP2toHRP3.h \
	/usr/local/include/bodyinfo_HRP2YH.h \
	/usr/local/include/bodyinfo_HRP2DOF7.h
