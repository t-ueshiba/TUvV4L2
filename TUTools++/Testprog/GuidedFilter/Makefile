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

INCDIRS		= -I../.. -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG
CFLAGS		= -O3
NVCCFLAGS	= -g
ifneq ($(findstring icpc,$(CXX)),)
#  CFLAGS	= -O3
  NVCCFLAGS	= -O			# must < -O2
#  CPPFLAGS     += -DSSE3
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
EXTHDRS		= ../../TU/Array++.h \
		../../TU/BoxFilter.h \
		../../TU/Filter2.h \
		../../TU/GuidedFilter.h \
		../../TU/functional.h \
		../../TU/iterator.h \
		../../TU/pair.h \
		../../TU/simd/allocator.h \
		../../TU/simd/arithmetic.h \
		../../TU/simd/arm/allocator.h \
		../../TU/simd/arm/arch.h \
		../../TU/simd/arm/arithmetic.h \
		../../TU/simd/arm/bit_shift.h \
		../../TU/simd/arm/cast.h \
		../../TU/simd/arm/compare.h \
		../../TU/simd/arm/cvt.h \
		../../TU/simd/arm/insert_extract.h \
		../../TU/simd/arm/load_store.h \
		../../TU/simd/arm/logical.h \
		../../TU/simd/arm/select.h \
		../../TU/simd/arm/shift.h \
		../../TU/simd/arm/type_traits.h \
		../../TU/simd/arm/vec.h \
		../../TU/simd/arm/zero.h \
		../../TU/simd/bit_shift.h \
		../../TU/simd/cast.h \
		../../TU/simd/compare.h \
		../../TU/simd/config.h \
		../../TU/simd/cvt.h \
		../../TU/simd/cvtdown_iterator.h \
		../../TU/simd/cvtup_iterator.h \
		../../TU/simd/insert_extract.h \
		../../TU/simd/intel/allocator.h \
		../../TU/simd/intel/arch.h \
		../../TU/simd/intel/arithmetic.h \
		../../TU/simd/intel/bit_shift.h \
		../../TU/simd/intel/cast.h \
		../../TU/simd/intel/compare.h \
		../../TU/simd/intel/cvt.h \
		../../TU/simd/intel/dup.h \
		../../TU/simd/intel/insert_extract.h \
		../../TU/simd/intel/load_store.h \
		../../TU/simd/intel/logical.h \
		../../TU/simd/intel/logical_base.h \
		../../TU/simd/intel/select.h \
		../../TU/simd/intel/shift.h \
		../../TU/simd/intel/shuffle.h \
		../../TU/simd/intel/svml.h \
		../../TU/simd/intel/type_traits.h \
		../../TU/simd/intel/unpack.h \
		../../TU/simd/intel/vec.h \
		../../TU/simd/intel/zero.h \
		../../TU/simd/load_iterator.h \
		../../TU/simd/load_store.h \
		../../TU/simd/logical.h \
		../../TU/simd/misc.h \
		../../TU/simd/row_vec_iterator.h \
		../../TU/simd/select.h \
		../../TU/simd/shift.h \
		../../TU/simd/shift_iterator.h \
		../../TU/simd/simd.h \
		../../TU/simd/store_iterator.h \
		../../TU/simd/transform.h \
		../../TU/simd/type_traits.h \
		../../TU/simd/vec.h \
		../../TU/simd/zero.h \
		../../TU/tuple.h \
		../../TU/types.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: ../../TU/types.h ../../TU/GuidedFilter.h ../../TU/BoxFilter.h \
	../../TU/Filter2.h ../../TU/iterator.h ../../TU/tuple.h \
	../../TU/functional.h ../../TU/pair.h ../../TU/Array++.h \
	../../TU/simd/simd.h ../../TU/simd/config.h ../../TU/simd/vec.h \
	../../TU/simd/type_traits.h ../../TU/simd/intel/type_traits.h \
	../../TU/simd/arm/type_traits.h ../../TU/simd/intel/vec.h \
	../../TU/simd/intel/arch.h ../../TU/simd/arm/vec.h \
	../../TU/simd/arm/arch.h ../../TU/simd/allocator.h \
	../../TU/simd/intel/allocator.h ../../TU/simd/arm/allocator.h \
	../../TU/simd/load_store.h ../../TU/simd/intel/load_store.h \
	../../TU/simd/arm/load_store.h ../../TU/simd/zero.h \
	../../TU/simd/intel/zero.h ../../TU/simd/arm/zero.h \
	../../TU/simd/cast.h ../../TU/simd/intel/cast.h \
	../../TU/simd/arm/cast.h ../../TU/simd/insert_extract.h \
	../../TU/simd/intel/insert_extract.h \
	../../TU/simd/arm/insert_extract.h ../../TU/simd/shift.h \
	../../TU/simd/intel/shift.h ../../TU/simd/arm/shift.h \
	../../TU/simd/bit_shift.h ../../TU/simd/intel/bit_shift.h \
	../../TU/simd/arm/bit_shift.h ../../TU/simd/cvt.h \
	../../TU/simd/intel/cvt.h ../../TU/simd/intel/dup.h \
	../../TU/simd/intel/unpack.h ../../TU/simd/arm/cvt.h \
	../../TU/simd/logical.h ../../TU/simd/intel/logical.h \
	../../TU/simd/intel/logical_base.h ../../TU/simd/arm/logical.h \
	../../TU/simd/compare.h ../../TU/simd/intel/compare.h \
	../../TU/simd/arm/compare.h ../../TU/simd/select.h \
	../../TU/simd/intel/select.h ../../TU/simd/arm/select.h \
	../../TU/simd/arithmetic.h ../../TU/simd/intel/arithmetic.h \
	../../TU/simd/arm/arithmetic.h ../../TU/simd/misc.h \
	../../TU/simd/intel/shuffle.h ../../TU/simd/intel/svml.h \
	../../TU/simd/transform.h ../../TU/simd/load_iterator.h \
	../../TU/simd/store_iterator.h ../../TU/simd/cvtdown_iterator.h \
	../../TU/simd/cvtup_iterator.h ../../TU/simd/shift_iterator.h \
	../../TU/simd/row_vec_iterator.h
