#
#  $Id: Makefile,v 1.76 2009-07-09 04:18:48 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include/TU
INCDIRS		= -I. -I$(PREFIX)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS     += -axP
  else
    CPPFLAGS   += -DSSSE3
    CFLAGS     += -xN
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		= /usr/local/include/TU/Allocator.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Bezier++.h \
		/usr/local/include/TU/BlockMatrix++.cc \
		/usr/local/include/TU/BlockMatrix++.h \
		/usr/local/include/TU/Camera.h \
		/usr/local/include/TU/CorrectIntensity.h \
		/usr/local/include/TU/EdgeDetector.h \
		/usr/local/include/TU/GaussianConvolver.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/IIRFilter.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Mapping.h \
		/usr/local/include/TU/Mesh++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Normalize.h \
		/usr/local/include/TU/Nurbs++.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/Random.h \
		/usr/local/include/TU/Serial.h \
		/usr/local/include/TU/Vector++.cc \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/Warp.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/utility.h
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
SRCS		= Bezier++.cc \
		BlockMatrix++.cc \
		BlockMatrix++.inst.cc \
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
		Mesh++.cc \
		Normalize.cc \
		Nurbs++.cc \
		Profiler.cc \
		Random.cc \
		Rotation.cc \
		Serial.cc \
		TriggerGenerator.cc \
		Vector++.cc \
		Vector++.inst.cc \
		Warp.cc \
		manipulators.cc
OBJS		= Bezier++.o \
		BlockMatrix++.o \
		BlockMatrix++.inst.o \
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
		Mesh++.o \
		Normalize.o \
		Nurbs++.o \
		Profiler.o \
		Random.o \
		Rotation.o \
		Serial.o \
		TriggerGenerator.o \
		Vector++.o \
		Vector++.inst.o \
		Warp.o \
		manipulators.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.76 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Bezier++.o: /usr/local/include/TU/Bezier++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
BlockMatrix++.o: /usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
BlockMatrix++.inst.o: /usr/local/include/TU/BlockMatrix++.cc \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
Camera.o: /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
CameraBase.o: /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
CameraWithDistortion.o: /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
CameraWithEuclideanImagePlane.o: /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
CameraWithFocalLength.o: /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
CanonicalCamera.o: /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
ConversionFromYUV.o: /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
CorrectIntensity.o: /usr/local/include/TU/CorrectIntensity.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/mmInstructions.h
EdgeDetector.o: /usr/local/include/TU/EdgeDetector.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/mmInstructions.h
GaussianCoefficients.o: /usr/local/include/TU/GaussianConvolver.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/IIRFilter.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Minimize.h
GenericImage.o: /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
Image++.inst.o: /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
ImageBase.o: /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Manip.h
ImageLine.o: /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h
Mapping.o: /usr/local/include/TU/Mapping.h /usr/local/include/TU/utility.h \
	/usr/local/include/TU/Normalize.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Minimize.h
Mesh++.o: /usr/local/include/TU/Mesh++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Allocator.h /usr/local/include/TU/List.h
Normalize.o: /usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
Nurbs++.o: /usr/local/include/TU/utility.h /usr/local/include/TU/Nurbs++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
Profiler.o: /usr/local/include/TU/Profiler.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
Random.o: /usr/local/include/TU/Random.h
Rotation.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
Serial.o: /usr/local/include/TU/Serial.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
TriggerGenerator.o: /usr/local/include/TU/Serial.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
Vector++.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
Vector++.inst.o: /usr/local/include/TU/Vector++.cc \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
Warp.o: /usr/local/include/TU/Warp.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/mmInstructions.h
