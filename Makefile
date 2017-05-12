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

INCDIRS		= -I.
CPPFLAGS	= -DNDEBUG
CFLAGS		= -O3 -Wall
NVCCFLAGS	= -O
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE4
  CFLAGS       += -msse4
endif
CCFLAGS		= $(CFLAGS)

LIBS		=
LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

OTHER_DIR	= $(HOME)/projects/HRP-5P/hrp5p-calib/src/TUTools++

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		=
HDRS		= TU/Array++.h \
		TU/BandMatrix++.h \
		TU/Bezier++.h \
		TU/BlockDiagonalMatrix++.h \
		TU/BoxFilter.h \
		TU/Camera++.h \
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
		TU/pair.h \
		TU/range.h \
		TU/simd/BufTraits.h \
		TU/simd/allocator.h \
		TU/simd/arithmetic.h \
		TU/simd/arm/allocator.h \
		TU/simd/arm/arch.h \
		TU/simd/arm/arithmetic.h \
		TU/simd/arm/bit_shift.h \
		TU/simd/arm/cast.h \
		TU/simd/arm/compare.h \
		TU/simd/arm/cvt.h \
		TU/simd/arm/dup.h \
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
		TU/simd/cvt_iterator.h \
		TU/simd/cvtdown_iterator.h \
		TU/simd/cvtup_iterator.h \
		TU/simd/dup.h \
		TU/simd/insert_extract.h \
		TU/simd/load_iterator.h \
		TU/simd/load_store.h \
		TU/simd/logical.h \
		TU/simd/lookup.h \
		TU/simd/misc.h \
		TU/simd/pack.h \
		TU/simd/select.h \
		TU/simd/shift.h \
		TU/simd/shift_iterator.h \
		TU/simd/simd.h \
		TU/simd/store_iterator.h \
		TU/simd/transform.h \
		TU/simd/type_traits.h \
		TU/simd/vec.h \
		TU/simd/x86/allocator.h \
		TU/simd/x86/arch.h \
		TU/simd/x86/arithmetic.h \
		TU/simd/x86/bit_shift.h \
		TU/simd/x86/cast.h \
		TU/simd/x86/compare.h \
		TU/simd/x86/cvt.h \
		TU/simd/x86/dup.h \
		TU/simd/x86/insert_extract.h \
		TU/simd/x86/load_store.h \
		TU/simd/x86/logical.h \
		TU/simd/x86/logical_base.h \
		TU/simd/x86/lookup.h \
		TU/simd/x86/misc.h \
		TU/simd/x86/select.h \
		TU/simd/x86/shift.h \
		TU/simd/x86/shuffle.h \
		TU/simd/x86/svml.h \
		TU/simd/x86/type_traits.h \
		TU/simd/x86/unpack.h \
		TU/simd/x86/vec.h \
		TU/simd/x86/zero.h \
		TU/simd/zero.h \
		TU/tuple.h \
		TU/types.h
SRCS		= ColorConverter.cc \
		EdgeDetector.cc \
		FIRGaussianCoefficients.cc \
		FeatureMatch.cc \
		GaussianCoefficients.cc \
		GenericImage.cc \
		ImageBase.cc \
		PM16C_04.cc \
		Random.cc \
		Rectify.cc \
		SHOT602.cc \
		SURFCreator.cc \
		Serial.cc \
		TriggerGenerator.cc \
		fdstream.cc \
		io.cc \
		manipulators.cc
OBJS		= ColorConverter.o \
		EdgeDetector.o \
		FIRGaussianCoefficients.o \
		FeatureMatch.o \
		GaussianCoefficients.o \
		GenericImage.o \
		ImageBase.o \
		PM16C_04.o \
		Random.o \
		Rectify.o \
		SHOT602.o \
		SURFCreator.o \
		Serial.o \
		TriggerGenerator.o \
		fdstream.o \
		io.o \
		manipulators.o

OTHER_HDRS	= TU/Array++.h \
		TU/BandMatrix++.h \
		TU/BlockDiagonalMatrix++.h \
		TU/BoxFilter.h \
		TU/Camera++.h \
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
		TU/GuidedFilter.h \
		TU/ICIA.h \
		TU/IIRFilter.h \
		TU/Image++.h \
		TU/IntegralImage.h \
		TU/List.h \
		TU/Manip.h \
		TU/Minimize.h \
		TU/Movie.h \
		TU/Profiler.h \
		TU/Random.h \
		TU/Ransac.h \
		TU/Rectify.h \
		TU/SADStereo.h \
		TU/SURFCreator.h \
		TU/SeparableFilter2.h \
		TU/SparseMatrix++.h \
		TU/StereoBase.h \
		TU/StereoUtility.h \
		TU/Vector++.h \
		TU/Warp.h \
		TU/algorithm.h \
		TU/fdstream.h \
		TU/functional.h \
		TU/io.h \
		TU/iterator.h \
		TU/pair.h \
		TU/range.h \
		TU/simd/BufTraits.h \
		TU/simd/allocator.h \
		TU/simd/arithmetic.h \
		TU/simd/arm/allocator.h \
		TU/simd/arm/arch.h \
		TU/simd/arm/arithmetic.h \
		TU/simd/arm/bit_shift.h \
		TU/simd/arm/cast.h \
		TU/simd/arm/compare.h \
		TU/simd/arm/cvt.h \
		TU/simd/arm/dup.h \
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
		TU/simd/cvt_iterator.h \
		TU/simd/cvtdown_iterator.h \
		TU/simd/cvtup_iterator.h \
		TU/simd/dup.h \
		TU/simd/insert_extract.h \
		TU/simd/load_iterator.h \
		TU/simd/load_store.h \
		TU/simd/logical.h \
		TU/simd/lookup.h \
		TU/simd/misc.h \
		TU/simd/pack.h \
		TU/simd/select.h \
		TU/simd/shift.h \
		TU/simd/shift_iterator.h \
		TU/simd/simd.h \
		TU/simd/store_iterator.h \
		TU/simd/transform.h \
		TU/simd/type_traits.h \
		TU/simd/vec.h \
		TU/simd/x86/allocator.h \
		TU/simd/x86/arch.h \
		TU/simd/x86/arithmetic.h \
		TU/simd/x86/bit_shift.h \
		TU/simd/x86/cast.h \
		TU/simd/x86/compare.h \
		TU/simd/x86/cvt.h \
		TU/simd/x86/dup.h \
		TU/simd/x86/insert_extract.h \
		TU/simd/x86/load_store.h \
		TU/simd/x86/logical.h \
		TU/simd/x86/logical_base.h \
		TU/simd/x86/lookup.h \
		TU/simd/x86/misc.h \
		TU/simd/x86/select.h \
		TU/simd/x86/shift.h \
		TU/simd/x86/shuffle.h \
		TU/simd/x86/svml.h \
		TU/simd/x86/type_traits.h \
		TU/simd/x86/unpack.h \
		TU/simd/x86/vec.h \
		TU/simd/x86/zero.h \
		TU/simd/zero.h \
		TU/tuple.h \
		TU/types.h
OTHER_SRCS	= ColorConverter.cc \
		EdgeDetector.cc \
		FIRGaussianCoefficients.cc \
		FeatureMatch.cc \
		GaussianCoefficients.cc \
		GenericImage.cc \
		ImageBase.cc \
		Random.cc \
		Rectify.cc \
		SURFCreator.cc \
		fdstream.cc \
		io.cc \
		manipulators.cc

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
include $(PROJECT)/lib/other.mk
###
ColorConverter.o: TU/Image++.h TU/types.h TU/pair.h TU/Vector++.h \
	TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h
EdgeDetector.o: TU/EdgeDetector.h TU/Image++.h TU/types.h TU/pair.h \
	TU/Vector++.h TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h \
	TU/iterator.h TU/Geometry++.h TU/Minimize.h TU/simd/simd.h \
	TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/x86/type_traits.h TU/simd/arm/type_traits.h TU/simd/x86/vec.h \
	TU/simd/x86/arch.h TU/simd/arm/vec.h TU/simd/arm/arch.h \
	TU/simd/allocator.h TU/simd/x86/allocator.h TU/simd/arm/allocator.h \
	TU/simd/load_store.h TU/simd/x86/load_store.h \
	TU/simd/arm/load_store.h TU/simd/zero.h TU/simd/x86/zero.h \
	TU/simd/arm/zero.h TU/simd/cast.h TU/simd/x86/cast.h \
	TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/x86/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/x86/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/x86/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/dup.h TU/simd/cvt.h TU/simd/x86/cvt.h TU/simd/x86/unpack.h \
	TU/simd/arm/cvt.h TU/simd/logical.h TU/simd/x86/logical.h \
	TU/simd/x86/logical_base.h TU/simd/arm/logical.h TU/simd/x86/dup.h \
	TU/simd/arm/dup.h TU/simd/compare.h TU/simd/x86/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/x86/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/x86/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/transform.h TU/functional.h \
	TU/simd/lookup.h TU/simd/x86/lookup.h TU/simd/arm/lookup.h \
	TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/shift_iterator.h TU/simd/BufTraits.h
FIRGaussianCoefficients.o: TU/FIRGaussianConvolver.h TU/FIRFilter.h \
	TU/SeparableFilter2.h TU/Array++.h TU/range.h TU/algorithm.h \
	TU/tuple.h TU/iterator.h
FeatureMatch.o: TU/FeatureMatch.h TU/Geometry++.h TU/Minimize.h \
	TU/Vector++.h TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h \
	TU/iterator.h TU/Random.h TU/types.h TU/Ransac.h TU/Manip.h
GaussianCoefficients.o: TU/GaussianConvolver.h TU/Vector++.h TU/Array++.h \
	TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h TU/IIRFilter.h \
	TU/SeparableFilter2.h TU/Minimize.h
GenericImage.o: TU/Image++.h TU/types.h TU/pair.h TU/Vector++.h \
	TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h
ImageBase.o: TU/Image++.h TU/types.h TU/pair.h TU/Vector++.h TU/Array++.h \
	TU/range.h TU/algorithm.h TU/tuple.h TU/iterator.h TU/Camera++.h \
	TU/Geometry++.h TU/Minimize.h TU/Manip.h
PM16C_04.o: TU/PM16C_04.h TU/Serial.h TU/fdstream.h TU/types.h TU/Manip.h
Random.o: TU/Random.h TU/types.h
Rectify.o: TU/Rectify.h TU/Warp.h TU/Image++.h TU/types.h TU/pair.h \
	TU/Vector++.h TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h \
	TU/iterator.h TU/Camera++.h TU/Geometry++.h TU/Minimize.h \
	TU/simd/simd.h TU/simd/config.h TU/simd/vec.h TU/simd/type_traits.h \
	TU/simd/x86/type_traits.h TU/simd/arm/type_traits.h TU/simd/x86/vec.h \
	TU/simd/x86/arch.h TU/simd/arm/vec.h TU/simd/arm/arch.h \
	TU/simd/allocator.h TU/simd/x86/allocator.h TU/simd/arm/allocator.h \
	TU/simd/load_store.h TU/simd/x86/load_store.h \
	TU/simd/arm/load_store.h TU/simd/zero.h TU/simd/x86/zero.h \
	TU/simd/arm/zero.h TU/simd/cast.h TU/simd/x86/cast.h \
	TU/simd/arm/cast.h TU/simd/insert_extract.h \
	TU/simd/x86/insert_extract.h TU/simd/arm/insert_extract.h \
	TU/simd/shift.h TU/simd/x86/shift.h TU/simd/arm/shift.h \
	TU/simd/bit_shift.h TU/simd/x86/bit_shift.h TU/simd/arm/bit_shift.h \
	TU/simd/dup.h TU/simd/cvt.h TU/simd/x86/cvt.h TU/simd/x86/unpack.h \
	TU/simd/arm/cvt.h TU/simd/logical.h TU/simd/x86/logical.h \
	TU/simd/x86/logical_base.h TU/simd/arm/logical.h TU/simd/x86/dup.h \
	TU/simd/arm/dup.h TU/simd/compare.h TU/simd/x86/compare.h \
	TU/simd/arm/compare.h TU/simd/select.h TU/simd/x86/select.h \
	TU/simd/arm/select.h TU/simd/arithmetic.h TU/simd/x86/arithmetic.h \
	TU/simd/arm/arithmetic.h TU/simd/transform.h TU/functional.h \
	TU/simd/lookup.h TU/simd/x86/lookup.h TU/simd/arm/lookup.h \
	TU/simd/load_iterator.h TU/simd/store_iterator.h \
	TU/simd/cvtdown_iterator.h TU/simd/cvtup_iterator.h \
	TU/simd/shift_iterator.h TU/simd/BufTraits.h
SHOT602.o: TU/SHOT602.h TU/Serial.h TU/fdstream.h TU/types.h TU/Manip.h
SURFCreator.o: TU/SURFCreator.h TU/Feature.h TU/Geometry++.h TU/Minimize.h \
	TU/Vector++.h TU/Array++.h TU/range.h TU/algorithm.h TU/tuple.h \
	TU/iterator.h TU/Manip.h TU/types.h TU/IntegralImage.h TU/Image++.h \
	TU/pair.h
Serial.o: TU/Serial.h TU/fdstream.h TU/types.h
TriggerGenerator.o: TU/TriggerGenerator.h TU/Serial.h TU/fdstream.h \
	TU/types.h TU/Manip.h
fdstream.o: TU/fdstream.h TU/types.h
io.o: TU/io.h
manipulators.o: TU/Manip.h TU/types.h
