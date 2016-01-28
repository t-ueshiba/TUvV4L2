#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
PROGRAM		= $(shell basename $(PWD))
#LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I../../.. -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE4
endif
CCFLAGS		= $(CFLAGS)

LIBS		=
#ifneq ($(findstring darwin,$(OSTYPE)),)
#  LIBS	       += -framework IOKit -framework CoreFoundation \
#		  -framework CoreServices
#endif

LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= /home/ueshiba/src/TUTools++/TU/functional.h \
		/home/ueshiba/src/TUTools++/TU/iterator.h \
		/home/ueshiba/src/TUTools++/TU/pair.h \
		/home/ueshiba/src/TUTools++/TU/simd/arm/arch.h \
		/home/ueshiba/src/TUTools++/TU/simd/arm/load_store.h \
		/home/ueshiba/src/TUTools++/TU/simd/arm/type_traits.h \
		/home/ueshiba/src/TUTools++/TU/simd/arm/vec.h \
		/home/ueshiba/src/TUTools++/TU/simd/config.h \
		/home/ueshiba/src/TUTools++/TU/simd/load_iterator.h \
		/home/ueshiba/src/TUTools++/TU/simd/load_store.h \
		/home/ueshiba/src/TUTools++/TU/simd/store_iterator.h \
		/home/ueshiba/src/TUTools++/TU/simd/type_traits.h \
		/home/ueshiba/src/TUTools++/TU/simd/vec.h \
		/home/ueshiba/src/TUTools++/TU/simd/x86/arch.h \
		/home/ueshiba/src/TUTools++/TU/simd/x86/load_store.h \
		/home/ueshiba/src/TUTools++/TU/simd/x86/type_traits.h \
		/home/ueshiba/src/TUTools++/TU/simd/x86/vec.h \
		/home/ueshiba/src/TUTools++/TU/tuple.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: /home/ueshiba/src/TUTools++/TU/simd/load_iterator.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/functional.h \
	/home/ueshiba/src/TUTools++/TU/pair.h \
	/home/ueshiba/src/TUTools++/TU/simd/load_store.h \
	/home/ueshiba/src/TUTools++/TU/simd/vec.h \
	/home/ueshiba/src/TUTools++/TU/simd/config.h \
	/home/ueshiba/src/TUTools++/TU/simd/type_traits.h \
	/home/ueshiba/src/TUTools++/TU/simd/x86/type_traits.h \
	/home/ueshiba/src/TUTools++/TU/simd/arm/type_traits.h \
	/home/ueshiba/src/TUTools++/TU/simd/x86/vec.h \
	/home/ueshiba/src/TUTools++/TU/simd/x86/arch.h \
	/home/ueshiba/src/TUTools++/TU/simd/arm/vec.h \
	/home/ueshiba/src/TUTools++/TU/simd/arm/arch.h \
	/home/ueshiba/src/TUTools++/TU/simd/x86/load_store.h \
	/home/ueshiba/src/TUTools++/TU/simd/arm/load_store.h \
	/home/ueshiba/src/TUTools++/TU/simd/store_iterator.h
