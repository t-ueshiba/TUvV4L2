#
#  $Id: Makefile,v 1.52 2008-06-16 02:21:04 ueshiba Exp $
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
    CFLAGS	= -O3 -axP -parallel -ip
  else
    CPPFLAGS   += -DSSE2
    CFLAGS	= -O3 -tpp7 -xW -ip
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= TU/Allocator++.h \
		TU/Bezier++.h \
		TU/BlockMatrix++.cc \
		TU/BlockMatrix++.h \
		TU/Geometry++.cc \
		TU/Geometry++.h \
		TU/Heap++.h \
		TU/Image++.cc \
		TU/Image++.h \
		TU/Manip.h \
		TU/Mesh++.h \
		TU/Nurbs++.h \
		TU/PSTree++.h \
		TU/Random.h \
		TU/Serial++.h \
		TU/TU/Array++.h \
		TU/TU/List++.h \
		TU/TU/Minimize++.h \
		TU/TU/TU/types.h \
		TU/TU/Vector++.h \
		TU/TU/utility.h \
		TU/Vector++.cc \
		TU/mmInstructions.h
HDRS		= Allocator++.h \
		Array++.h \
		Bezier++.h \
		BlockMatrix++.h \
		Geometry++.h \
		Heap++.h \
		Image++.ORG.h \
		Image++.h \
		List++.h \
		Manip.h \
		Mesh++.h \
		Minimize++.h \
		Nurbs++.h \
		PSTree++.h \
		Random.h \
		Ransac++.h \
		Serial++.h \
		Vector++.h \
		mmInstructions.h \
		types.h \
		utility.h
SRCS		= Allocator++.cc \
		Bezier++.cc \
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
		GenericImage.cc \
		Geometry++.cc \
		Geometry++.inst.cc \
		Heap++.cc \
		IIRFilter.cc \
		Image++.cc \
		Image++.inst.cc \
		ImageBase.cc \
		ImageLine.cc \
		LinearMapping.cc \
		List++.cc \
		Mesh++.cc \
		Microscope.cc \
		Normalize.cc \
		Nurbs++.cc \
		PSTree++.cc \
		Pata.cc \
		Puma.cc \
		Random.cc \
		Rotation.cc \
		Serial.cc \
		TUTools++.sa.cc \
		TriggerGenerator.cc \
		Vector++.cc \
		Vector++.inst.cc \
		Warp.cc \
		manipulators.cc
OBJS		= Allocator++.o \
		Bezier++.o \
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
		GenericImage.o \
		Geometry++.o \
		Geometry++.inst.o \
		Heap++.o \
		IIRFilter.o \
		Image++.o \
		Image++.inst.o \
		ImageBase.o \
		ImageLine.o \
		LinearMapping.o \
		List++.o \
		Mesh++.o \
		Microscope.o \
		Normalize.o \
		Nurbs++.o \
		PSTree++.o \
		Pata.o \
		Puma.o \
		Random.o \
		Rotation.o \
		Serial.o \
		TUTools++.sa.o \
		TriggerGenerator.o \
		Vector++.o \
		Vector++.inst.o \
		Warp.o \
		manipulators.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.52 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Allocator++.o: TU/Allocator++.h TU/TU/List++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
Bezier++.o: TU/Bezier++.h TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
BlockMatrix++.o: TU/BlockMatrix++.h TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
BlockMatrix++.inst.o: TU/BlockMatrix++.cc TU/BlockMatrix++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
Camera.o: TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
CameraBase.o: TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
CameraWithDistortion.o: TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
CameraWithEuclideanImagePlane.o: TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
CameraWithFocalLength.o: TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
CanonicalCamera.o: TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
ConversionFromYUV.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
CorrectIntensity.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/mmInstructions.h
EdgeDetector.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/mmInstructions.h
GenericImage.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
Geometry++.o: TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
Geometry++.inst.o: TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h TU/Geometry++.cc
Heap++.o: TU/Heap++.h TU/TU/Array++.h TU/TU/TU/types.h
IIRFilter.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/Image++.cc TU/mmInstructions.h
Image++.o: TU/TU/utility.h TU/Image++.h TU/Geometry++.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h TU/mmInstructions.h
Image++.inst.o: TU/Image++.cc TU/TU/utility.h TU/Image++.h TU/Geometry++.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/mmInstructions.h
ImageBase.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/Manip.h
ImageLine.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
LinearMapping.o: TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
List++.o: TU/TU/List++.h
Mesh++.o: TU/Mesh++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/Allocator++.h TU/TU/List++.h
Microscope.o: TU/Serial++.h TU/Manip.h TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
Normalize.o: TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
Nurbs++.o: TU/TU/utility.h TU/Nurbs++.h TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
PSTree++.o: TU/PSTree++.h TU/Heap++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/TU/List++.h
Pata.o: TU/Serial++.h TU/Manip.h TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
Puma.o: TU/Serial++.h TU/Manip.h TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
Random.o: TU/Random.h
Rotation.o: TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
Serial.o: TU/Serial++.h TU/Manip.h TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
TUTools++.sa.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h \
	TU/TU/Minimize++.h TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/Serial++.h TU/Manip.h
TriggerGenerator.o: TU/Serial++.h TU/Manip.h TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h
Vector++.o: TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
Vector++.inst.o: TU/Vector++.cc TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
Warp.o: TU/Image++.h TU/Geometry++.h TU/TU/utility.h TU/TU/Minimize++.h \
	TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h TU/mmInstructions.h \
	TU/Image++.cc
