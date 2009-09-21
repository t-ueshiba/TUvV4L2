#
#  $Id: Makefile,v 1.84 2009-09-21 07:09:05 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/TU
INCDIRS		= -I.

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
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
EXTHDRS		= TU/BlockMatrix++.h \
		TU/Camera.h \
		TU/CorrectIntensity.h \
		TU/EdgeDetector.h \
		TU/GaussianConvolver.h \
		TU/Image++.h \
		TU/Manip.h \
		TU/Mapping.h \
		TU/Minimize.h \
		TU/Profiler.h \
		TU/Random.h \
		TU/Serial.h \
		TU/TU/Geometry++.h \
		TU/TU/IIRFilter.h \
		TU/TU/TU/Array++.h \
		TU/TU/TU/Normalize.h \
		TU/TU/TU/TU/types.h \
		TU/TU/Vector++.h \
		TU/TU/utility.h \
		TU/Warp.h \
		TU/mmInstructions.h \
		TU/windows/fakeWindows.h
HDRS		= Allocator.h \
		Array++.h \
		Bezier++.h \
		BlockMatrix++.h \
		Camera.h \
		CorrectIntensity.h \
		DericheConvolver.h \
		EdgeDetector.h \
		GaussianConvolver.h \
		Geometry++.h \
		Heap.h \
		IIRFilter.h \
		Image++.h \
		IntegralImage.h \
		List.h \
		Manip.h \
		Mapping.h \
		Mesh++.h \
		Minimize.h \
		Movie.h \
		Normalize.h \
		Nurbs++.h \
		PSTree.h \
		Profiler.h \
		Random.h \
		Ransac.h \
		Serial.h \
		Vector++.h \
		Warp.h \
		mmInstructions.h \
		types.h \
		utility.h
SRCS		= BlockMatrix++.inst.cc \
		Camera.cc \
		CameraBase.cc \
		CameraWithDistortion.cc \
		CameraWithEuclideanImagePlane.cc \
		CameraWithFocalLength.cc \
		CanonicalCamera.cc \
		ConversionFromYUV.cc \
		CorrectIntensity.cc \
		EdgeDetector.cc \
		GaussianCoefficients.cc \
		GenericImage.cc \
		Image++.inst.cc \
		ImageBase.cc \
		ImageLine.cc \
		Mapping.cc \
		Normalize.cc \
		Profiler.cc \
		Random.cc \
		Rotation.cc \
		Serial.cc \
		TriggerGenerator.cc \
		Vector++.inst.cc \
		Warp.cc \
		manipulators.cc
OBJS		= BlockMatrix++.inst.o \
		Camera.o \
		CameraBase.o \
		CameraWithDistortion.o \
		CameraWithEuclideanImagePlane.o \
		CameraWithFocalLength.o \
		CanonicalCamera.o \
		ConversionFromYUV.o \
		CorrectIntensity.o \
		EdgeDetector.o \
		GaussianCoefficients.o \
		GenericImage.o \
		Image++.inst.o \
		ImageBase.o \
		ImageLine.o \
		Mapping.o \
		Normalize.o \
		Profiler.o \
		Random.o \
		Rotation.o \
		Serial.o \
		TriggerGenerator.o \
		Vector++.inst.o \
		Warp.o \
		manipulators.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.84 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
BlockMatrix++.inst.o: TU/BlockMatrix++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h
Camera.o: TU/Camera.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
CameraBase.o: TU/Camera.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
CameraWithDistortion.o: TU/Camera.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
CameraWithEuclideanImagePlane.o: TU/Camera.h TU/TU/Geometry++.h \
	TU/TU/Vector++.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h \
	TU/TU/TU/Normalize.h
CameraWithFocalLength.o: TU/Camera.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
CanonicalCamera.o: TU/Camera.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
ConversionFromYUV.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
CorrectIntensity.o: TU/CorrectIntensity.h TU/Image++.h TU/TU/Geometry++.h \
	TU/TU/Vector++.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h \
	TU/TU/TU/Normalize.h TU/mmInstructions.h
EdgeDetector.o: TU/EdgeDetector.h TU/Image++.h TU/TU/Geometry++.h \
	TU/TU/Vector++.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h \
	TU/TU/TU/Normalize.h TU/mmInstructions.h
GaussianCoefficients.o: TU/GaussianConvolver.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/IIRFilter.h \
	TU/mmInstructions.h TU/Minimize.h
GenericImage.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
Image++.inst.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
ImageBase.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h \
	TU/Camera.h TU/Manip.h
ImageLine.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h
Mapping.o: TU/Mapping.h TU/TU/utility.h TU/TU/TU/Normalize.h \
	TU/TU/Vector++.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/Minimize.h
Normalize.o: TU/TU/TU/Normalize.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h
Profiler.o: TU/Profiler.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h \
	TU/windows/fakeWindows.h
Random.o: TU/Random.h TU/TU/TU/TU/types.h TU/windows/fakeWindows.h
Rotation.o: TU/TU/Vector++.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h
Serial.o: TU/Serial.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h
TriggerGenerator.o: TU/Serial.h TU/TU/Vector++.h TU/TU/TU/Array++.h \
	TU/TU/TU/TU/types.h
Vector++.inst.o: TU/TU/Vector++.h TU/TU/TU/Array++.h TU/TU/TU/TU/types.h
Warp.o: TU/Warp.h TU/Image++.h TU/TU/Geometry++.h TU/TU/Vector++.h \
	TU/TU/TU/Array++.h TU/TU/TU/TU/types.h TU/TU/TU/Normalize.h \
	TU/Camera.h TU/mmInstructions.h
manipulators.o: TU/Manip.h TU/TU/TU/TU/types.h
