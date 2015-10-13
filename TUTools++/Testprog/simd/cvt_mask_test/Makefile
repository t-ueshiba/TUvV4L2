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
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DAVX2
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
EXTHDRS		= ../../../TU/functional.h \
		../../../TU/simd/arm/arch.h \
		../../../TU/simd/arm/bit_shift.h \
		../../../TU/simd/arm/cast.h \
		../../../TU/simd/arm/compare.h \
		../../../TU/simd/arm/cvt.h \
		../../../TU/simd/arm/cvt_mask.h \
		../../../TU/simd/arm/load_store.h \
		../../../TU/simd/arm/type_traits.h \
		../../../TU/simd/arm/vec.h \
		../../../TU/simd/arm/zero.h \
		../../../TU/simd/bit_shift.h \
		../../../TU/simd/cast.h \
		../../../TU/simd/compare.h \
		../../../TU/simd/config.h \
		../../../TU/simd/cvt.h \
		../../../TU/simd/cvt_mask.h \
		../../../TU/simd/intel/arch.h \
		../../../TU/simd/intel/bit_shift.h \
		../../../TU/simd/intel/cast.h \
		../../../TU/simd/intel/compare.h \
		../../../TU/simd/intel/cvt.h \
		../../../TU/simd/intel/cvt_mask.h \
		../../../TU/simd/intel/dup.h \
		../../../TU/simd/intel/load_store.h \
		../../../TU/simd/intel/type_traits.h \
		../../../TU/simd/intel/unpack.h \
		../../../TU/simd/intel/vec.h \
		../../../TU/simd/intel/zero.h \
		../../../TU/simd/load_store.h \
		../../../TU/simd/type_traits.h \
		../../../TU/simd/vec.h \
		../../../TU/simd/zero.h \
		../../../TU/tuple.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: ../../../TU/simd/cvt_mask.h ../../../TU/simd/cvt.h \
	../../../TU/simd/zero.h ../../../TU/simd/vec.h ../../../TU/tuple.h \
	../../../TU/functional.h ../../../TU/simd/config.h \
	../../../TU/simd/type_traits.h ../../../TU/simd/intel/type_traits.h \
	../../../TU/simd/arm/type_traits.h ../../../TU/simd/intel/vec.h \
	../../../TU/simd/intel/arch.h ../../../TU/simd/arm/vec.h \
	../../../TU/simd/arm/arch.h ../../../TU/simd/intel/zero.h \
	../../../TU/simd/arm/zero.h ../../../TU/simd/cast.h \
	../../../TU/simd/intel/cast.h ../../../TU/simd/arm/cast.h \
	../../../TU/simd/bit_shift.h ../../../TU/simd/intel/bit_shift.h \
	../../../TU/simd/arm/bit_shift.h ../../../TU/simd/intel/cvt.h \
	../../../TU/simd/intel/dup.h ../../../TU/simd/intel/unpack.h \
	../../../TU/simd/arm/cvt.h ../../../TU/simd/intel/cvt_mask.h \
	../../../TU/simd/arm/cvt_mask.h ../../../TU/simd/load_store.h \
	../../../TU/simd/intel/load_store.h ../../../TU/simd/arm/load_store.h \
	../../../TU/simd/compare.h ../../../TU/simd/intel/compare.h \
	../../../TU/simd/arm/compare.h
