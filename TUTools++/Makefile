#
#  $Id: Makefile,v 1.69 2009-03-09 05:12:32 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I. -I$(HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g
ifeq ($(CCC), icpc)
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS	= -O3 -axP -ip
  else
    CPPFLAGS   += -DSSSE3
    CFLAGS	= -O3 -xN -ip
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Allocator.h \
		/home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Bezier++.h \
		/home/ueshiba/include/TU/BlockMatrix++.cc \
		/home/ueshiba/include/TU/BlockMatrix++.h \
		/home/ueshiba/include/TU/Camera.h \
		/home/ueshiba/include/TU/CorrectIntensity.h \
		/home/ueshiba/include/TU/EdgeDetector.h \
		/home/ueshiba/include/TU/GaussianConvolver.h \
		/home/ueshiba/include/TU/Geometry++.h \
		/home/ueshiba/include/TU/IIRFilter.h \
		/home/ueshiba/include/TU/Image++.h \
		/home/ueshiba/include/TU/List.h \
		/home/ueshiba/include/TU/Manip.h \
		/home/ueshiba/include/TU/Mapping.h \
		/home/ueshiba/include/TU/Mesh++.h \
		/home/ueshiba/include/TU/Minimize.h \
		/home/ueshiba/include/TU/Normalize.h \
		/home/ueshiba/include/TU/Nurbs++.h \
		/home/ueshiba/include/TU/Profiler.h \
		/home/ueshiba/include/TU/Random.h \
		/home/ueshiba/include/TU/Serial.h \
		/home/ueshiba/include/TU/Vector++.cc \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/Warp.h \
		/home/ueshiba/include/TU/mmInstructions.h \
		/home/ueshiba/include/TU/types.h \
		/home/ueshiba/include/TU/utility.h
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
		Microscope.cc \
		Normalize.cc \
		Nurbs++.cc \
		Pata.cc \
		Profiler.cc \
		Puma.cc \
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
		Microscope.o \
		Normalize.o \
		Nurbs++.o \
		Pata.o \
		Profiler.o \
		Puma.o \
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
REV		= $(shell echo $Revision: 1.69 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Bezier++.o: /home/ueshiba/include/TU/Bezier++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
BlockMatrix++.o: /home/ueshiba/include/TU/BlockMatrix++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
BlockMatrix++.inst.o: /home/ueshiba/include/TU/BlockMatrix++.cc \
	/home/ueshiba/include/TU/BlockMatrix++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Camera.o: /home/ueshiba/include/TU/Camera.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
CameraBase.o: /home/ueshiba/include/TU/Camera.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
CameraWithDistortion.o: /home/ueshiba/include/TU/Camera.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
CameraWithEuclideanImagePlane.o: /home/ueshiba/include/TU/Camera.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
CameraWithFocalLength.o: /home/ueshiba/include/TU/Camera.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
CanonicalCamera.o: /home/ueshiba/include/TU/Camera.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
ConversionFromYUV.o: /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
CorrectIntensity.o: /home/ueshiba/include/TU/CorrectIntensity.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h \
	/home/ueshiba/include/TU/mmInstructions.h
EdgeDetector.o: /home/ueshiba/include/TU/EdgeDetector.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h \
	/home/ueshiba/include/TU/mmInstructions.h
GaussianCoefficients.o: /home/ueshiba/include/TU/GaussianConvolver.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/IIRFilter.h \
	/home/ueshiba/include/TU/mmInstructions.h \
	/home/ueshiba/include/TU/Minimize.h
GenericImage.o: /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
Image++.inst.o: /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
ImageBase.o: /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h \
	/home/ueshiba/include/TU/Camera.h /home/ueshiba/include/TU/Manip.h
ImageLine.o: /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h
Mapping.o: /home/ueshiba/include/TU/Mapping.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Normalize.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Minimize.h
Mesh++.o: /home/ueshiba/include/TU/Mesh++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Allocator.h /home/ueshiba/include/TU/List.h
Microscope.o: /home/ueshiba/include/TU/Serial.h \
	/home/ueshiba/include/TU/Manip.h /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Normalize.o: /home/ueshiba/include/TU/Normalize.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Nurbs++.o: /home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Nurbs++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Pata.o: /home/ueshiba/include/TU/Serial.h /home/ueshiba/include/TU/Manip.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Profiler.o: /home/ueshiba/include/TU/Profiler.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Puma.o: /home/ueshiba/include/TU/Serial.h /home/ueshiba/include/TU/Manip.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Random.o: /home/ueshiba/include/TU/Random.h
Rotation.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Serial.o: /home/ueshiba/include/TU/Serial.h \
	/home/ueshiba/include/TU/Manip.h /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
TriggerGenerator.o: /home/ueshiba/include/TU/Serial.h \
	/home/ueshiba/include/TU/Manip.h /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Vector++.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Vector++.inst.o: /home/ueshiba/include/TU/Vector++.cc \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Warp.o: /home/ueshiba/include/TU/Warp.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Normalize.h \
	/home/ueshiba/include/TU/Camera.h \
	/home/ueshiba/include/TU/mmInstructions.h
