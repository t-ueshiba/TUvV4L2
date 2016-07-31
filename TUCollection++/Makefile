#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
#PROGRAM		= $(shell basename $(PWD))
LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I. -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG
CFLAGS		= -O3
NVCCFLAGS	= -O
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		=
LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= /usr/local/include/TU/Object++.h \
		/usr/local/include/TU/types.h
HDRS		= TU/Collection++.h
SRCS		= Collection++.cc \
		ObjTreeBase.cc \
		TUCollection++.sa.cc
OBJS		= Collection++.o \
		ObjTreeBase.o \
		TUCollection++.sa.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
Collection++.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
ObjTreeBase.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
TUCollection++.sa.o: TU/Collection++.h /usr/local/include/TU/Object++.h \
	/usr/local/include/TU/types.h
