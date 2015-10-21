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
EXTHDRS		= windows/fakeWindows.h
HDRS		= TU/Array++.h \
		TU/BandMatrix++.h \
		TU/Bezier++.h \
		TU/BlockDiagonalMatrix++.h \
		TU/BoxFilter.h \
		TU/CCSImage.h \
		TU/Camera++.h \
		TU/ComplexImage.h \
		TU/CorrectIntensity.h \
		TU/DP.h \
		TU/DericheConvolver.h \
		TU/EdgeDetector.h \
		TU/FIRFilter.h \
		TU/FIRGaussianConvolver.h \
		TU/Feature.h \
		TU/FeatureMatch.h \
		TU/Filter2.h \
		TU/GFStereo.h \
		TU/GaussianConvolver.h \
		TU/Geometry++.h \
		TU/GraphCuts.h \
		TU/GuidedFilter.h \
		TU/Heap.h \
		TU/ICIA.h \
		TU/IIRFilter.h \
		TU/Image++.h \
		TU/IntegralImage.h \
		TU/List.h \
		TU/Manip.h \
		TU/Mesh++.h \
		TU/Minimize.h \
		TU/Movie.h \
		TU/NDTree++.h \
		TU/Nurbs++.h \
		TU/PM16C_04.h \
		TU/PSTree.h \
		TU/Profiler.h \
		TU/Quantizer.h \
		TU/Random.h \
		TU/Ransac.h \
		TU/Rectify.h \
		TU/SADStereo.h \
		TU/SHOT602.h \
		TU/SURFCreator.h \
		TU/SeparableFilter2.h \
		TU/Serial.h \
		TU/SparseMatrix++.h \
		TU/StereoBase.h \
		TU/StereoIterators.h \
		TU/StereoUtility.h \
		TU/TreeFilter.h \
		TU/TriggerGenerator.h \
		TU/Vector++.h \
		TU/Warp.h \
		TU/WeightedMedianFilter.h \
		TU/algorithm.h \
		TU/fdstream.h \
		TU/functional.h \
		TU/io.h \
		TU/iterator.h \
		TU/simd/allocator.h \
		TU/simd/arithmetic.h \
		TU/simd/arm/allocator.h \
		TU/simd/arm/arch.h \
		TU/simd/arm/arithmetic.h \
		TU/simd/arm/bit_shift.h \
		TU/simd/arm/cast.h \
		TU/simd/arm/compare.h \
		TU/simd/arm/cvt.h \
		TU/simd/arm/cvt_mask.h \
		TU/simd/arm/insert_extract.h \
		TU/simd/arm/load_store.h \
		TU/simd/arm/logical.h \
		TU/simd/arm/lookup.h \
		TU/simd/arm/misc.h \
		TU/simd/arm/select.h \
		TU/simd/arm/shift.h \
		TU/simd/arm/type_traits.h \
		TU/simd/arm/vec.h \
		TU/simd/arm/zero.h \
		TU/simd/bit_shift.h \
		TU/simd/cast.h \
		TU/simd/compare.h \
		TU/simd/config.h \
		TU/simd/cvt.h \
		TU/simd/cvt_mask.h \
		TU/simd/cvtdown_iterator.h \
		TU/simd/cvtdown_mask_iterator.h \
		TU/simd/cvtup_iterator.h \
		TU/simd/cvtup_mask_iterator.h \
		TU/simd/insert_extract.h \
		TU/simd/intel/allocator.h \
		TU/simd/intel/arch.h \
		TU/simd/intel/arithmetic.h \
		TU/simd/intel/bit_shift.h \
		TU/simd/intel/cast.h \
		TU/simd/intel/compare.h \
		TU/simd/intel/cvt.h \
		TU/simd/intel/cvt_mask.h \
		TU/simd/intel/dup.h \
		TU/simd/intel/insert_extract.h \
		TU/simd/intel/load_store.h \
		TU/simd/intel/logical.h \
		TU/simd/intel/logical_base.h \
		TU/simd/intel/lookup.h \
		TU/simd/intel/misc.h \
		TU/simd/intel/select.h \
		TU/simd/intel/shift.h \
		TU/simd/intel/shuffle.h \
		TU/simd/intel/svml.h \
		TU/simd/intel/type_traits.h \
		TU/simd/intel/unpack.h \
		TU/simd/intel/vec.h \
		TU/simd/intel/zero.h \
		TU/simd/load_iterator.h \
		TU/simd/load_store.h \
		TU/simd/logical.h \
		TU/simd/lookup.h \
		TU/simd/misc.h \
		TU/simd/row_vec_iterator.h \
		TU/simd/select.h \
		TU/simd/shift.h \
		TU/simd/shift_iterator.h \
		TU/simd/simd.h \
		TU/simd/store_iterator.h \
		TU/simd/type_traits.h \
		TU/simd/vec.h \
		TU/simd/zero.h \
		TU/tuple.h \
		TU/types.h
SRCS		= BlockDiagonalMatrix++.inst.cc \
		ConversionFromYUV.cc \
		CorrectIntensity.cc \
		EdgeDetector.cc \
		FIRGaussianCoefficients.cc \
		FeatureMatch.cc \
		GaussianCoefficients.cc \
		GenericImage.cc \
		Image++.inst.cc \
		ImageBase.cc \
		ImageLine.cc \
		PM16C_04.cc \
		Profiler.cc \
		Random.cc \
		Rectify.cc \
		SHOT602.cc \
		SURFCreator.cc \
		Serial.cc \
		TriggerGenerator.cc \
		Vector++.inst.cc \
		Warp.cc \
		fdstream.cc \
		io.cc \
		manipulators.cc
OBJS		= BlockDiagonalMatrix++.inst.o \
		ConversionFromYUV.o \
		CorrectIntensity.o \
		EdgeDetector.o \
		FIRGaussianCoefficients.o \
		FeatureMatch.o \
		GaussianCoefficients.o \
		GenericImage.o \
		Image++.inst.o \
		ImageBase.o \
		ImageLine.o \
		PM16C_04.o \
		Profiler.o \
		Random.o \
		Rectify.o \
		SHOT602.o \
		SURFCreator.o \
		Serial.o \
		TriggerGenerator.o \
		Vector++.inst.o \
		Warp.o \
		fdstream.o \
		io.o \
		manipulators.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
BlockDiagonalMatrix++.inst.o: TU/BlockDiagonalMatrix++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h
ConversionFromYUV.o: TU/Image++.h TU/types.h TU/Geometry++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h
CorrectIntensity.o: TU/CorrectIntensity.h TU/Image++.h TU/types.h \
	TU/Geometry++.h TU/Vector++.h TU/Array++.h TU/iterator.h TU/tuple.h \
	TU/functional.h TU/simd/simd.h TU/simd/config.h TU/simd/vec.h \
	TU/simd/type_traits.h TU/simd/intel/type_traits.h \
	TU/simd/arm/type_traits.h TU/simd/intel/vec.h TU/simd/intel/arch.h \
	TU/simd/arm/vec.h TU/simd/arm/arch.h TU/simd/allocator.h \
	TU/simd/intel/allocator.h TU/simd/arm/allocator.h \
	TU/simd/load_store.h TU/simd/intel/load_store.h \
	TU/simd/arm/load_store.h TU/simd/zero.h TU/simd/intel/zero.h \
	TU/simd/arm/zero.h TU/simd/cast.h TU/simd/intel/cast.h \
	TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h
EdgeDetector.o: TU/EdgeDetector.h TU/Image++.h TU/types.h TU/Geometry++.h \
	TU/Vector++.h TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h \
	TU/simd/simd.h TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h
FIRGaussianCoefficients.o: TU/FIRGaussianConvolver.h TU/FIRFilter.h \
	TU/SeparableFilter2.h TU/Array++.h TU/iterator.h TU/tuple.h \
	TU/functional.h TU/simd/simd.h TU/simd/config.h TU/simd/vec.h \
	TU/simd/type_traits.h TU/simd/intel/type_traits.h \
	TU/simd/arm/type_traits.h TU/simd/intel/vec.h TU/simd/intel/arch.h \
	TU/simd/arm/vec.h TU/simd/arm/arch.h TU/simd/allocator.h \
	TU/simd/intel/allocator.h TU/simd/arm/allocator.h \
	TU/simd/load_store.h TU/simd/intel/load_store.h \
	TU/simd/arm/load_store.h TU/simd/zero.h TU/simd/intel/zero.h \
	TU/simd/arm/zero.h TU/simd/cast.h TU/simd/intel/cast.h \
	TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h
FeatureMatch.o: TU/FeatureMatch.h TU/Geometry++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h \
	TU/Random.h TU/types.h TU/Ransac.h TU/Manip.h
GaussianCoefficients.o: TU/GaussianConvolver.h TU/Vector++.h TU/Array++.h \
	TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/IIRFilter.h \
	TU/SeparableFilter2.h TU/Minimize.h
GenericImage.o: TU/Image++.h TU/types.h TU/Geometry++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h
Image++.inst.o: TU/Image++.h TU/types.h TU/Geometry++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h
ImageBase.o: TU/Image++.h TU/types.h TU/Geometry++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h \
	TU/Camera++.h TU/Manip.h
ImageLine.o: TU/Image++.h TU/types.h TU/Geometry++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h
PM16C_04.o: TU/PM16C_04.h TU/Serial.h TU/fdstream.h TU/types.h TU/Manip.h
Profiler.o: TU/Profiler.h TU/types.h TU/Array++.h TU/iterator.h TU/tuple.h \
	TU/functional.h TU/simd/simd.h TU/simd/config.h TU/simd/vec.h \
	TU/simd/type_traits.h TU/simd/intel/type_traits.h \
	TU/simd/arm/type_traits.h TU/simd/intel/vec.h TU/simd/intel/arch.h \
	TU/simd/arm/vec.h TU/simd/arm/arch.h TU/simd/allocator.h \
	TU/simd/intel/allocator.h TU/simd/arm/allocator.h \
	TU/simd/load_store.h TU/simd/intel/load_store.h \
	TU/simd/arm/load_store.h TU/simd/zero.h TU/simd/intel/zero.h \
	TU/simd/arm/zero.h TU/simd/cast.h TU/simd/intel/cast.h \
	TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h \
	windows/fakeWindows.h
Random.o: TU/Random.h TU/types.h windows/fakeWindows.h
Rectify.o: TU/Rectify.h TU/Warp.h TU/Image++.h TU/types.h TU/Geometry++.h \
	TU/Vector++.h TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h \
	TU/simd/simd.h TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h \
	TU/Camera++.h TU/algorithm.h
SHOT602.o: TU/SHOT602.h TU/Serial.h TU/fdstream.h TU/types.h TU/Manip.h
SURFCreator.o: TU/SURFCreator.h TU/Feature.h TU/Geometry++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h \
	TU/Manip.h TU/types.h TU/IntegralImage.h TU/Image++.h TU/Heap.h
Serial.o: TU/Serial.h TU/fdstream.h TU/types.h
TriggerGenerator.o: TU/TriggerGenerator.h TU/Serial.h TU/fdstream.h \
	TU/types.h TU/Manip.h
Vector++.inst.o: TU/Vector++.h TU/Array++.h TU/iterator.h TU/tuple.h \
	TU/functional.h TU/simd/simd.h TU/simd/config.h TU/simd/vec.h \
	TU/simd/type_traits.h TU/simd/intel/type_traits.h \
	TU/simd/arm/type_traits.h TU/simd/intel/vec.h TU/simd/intel/arch.h \
	TU/simd/arm/vec.h TU/simd/arm/arch.h TU/simd/allocator.h \
	TU/simd/intel/allocator.h TU/simd/arm/allocator.h \
	TU/simd/load_store.h TU/simd/intel/load_store.h \
	TU/simd/arm/load_store.h TU/simd/zero.h TU/simd/intel/zero.h \
	TU/simd/arm/zero.h TU/simd/cast.h TU/simd/intel/cast.h \
	TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h
Warp.o: TU/Warp.h TU/Image++.h TU/types.h TU/Geometry++.h TU/Vector++.h \
	TU/Array++.h TU/iterator.h TU/tuple.h TU/functional.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/intel/type_traits.h TU/simd/arm/type_traits.h \
	TU/simd/intel/vec.h TU/simd/intel/arch.h TU/simd/arm/vec.h \
	TU/simd/arm/arch.h TU/simd/allocator.h TU/simd/intel/allocator.h \
	TU/simd/arm/allocator.h TU/simd/load_store.h \
	TU/simd/intel/load_store.h TU/simd/arm/load_store.h TU/simd/zero.h \
	TU/simd/intel/zero.h TU/simd/arm/zero.h TU/simd/cast.h \
	TU/simd/intel/cast.h TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/intel/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/intel/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/intel/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/cvt.h TU/simd/intel/cvt.h TU/simd/intel/dup.h \
	TU/simd/intel/unpack.h TU/simd/arm/cvt.h TU/simd/cvt_mask.h \
	TU/simd/intel/cvt_mask.h TU/simd/arm/cvt_mask.h TU/simd/logical.h \
	TU/simd/intel/logical.h TU/simd/intel/logical_base.h \
	TU/simd/arm/logical.h TU/simd/compare.h TU/simd/intel/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/intel/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/intel/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/misc.h TU/simd/intel/shuffle.h \
	TU/simd/intel/svml.h TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/cvtdown_mask_iterator.h TU/simd/cvtup_mask_iterator.h \
	TU/simd/shift_iterator.h TU/simd/row_vec_iterator.h TU/Minimize.h \
	TU/Camera++.h
fdstream.o: TU/fdstream.h TU/types.h
io.o: TU/io.h
manipulators.o: TU/Manip.h TU/types.h
