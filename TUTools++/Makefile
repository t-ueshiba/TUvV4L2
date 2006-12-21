#
#  $Id: Makefile,v 1.32 2006-12-21 05:12:00 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(INCDIR)

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -g
ifeq ($(CCC), icpc)
  CPPFLAGS     += -DSSE3
  CCFLAGS	= -O3 -parallel
endif
LDFLAGS		= $(CCFLAGS)
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= TU/Allocator++.h \
		TU/Array++.cc \
		TU/Bezier++.h \
		TU/BlockMatrix++.cc \
		TU/BlockMatrix++.h \
		TU/Geometry++.cc \
		TU/Heap++.h \
		TU/Image++.cc \
		TU/Image++.h \
		TU/Manip.h \
		TU/Mesh++.h \
		TU/Minimize++.h \
		TU/Nurbs++.h \
		TU/PSTree++.h \
		TU/Random.h \
		TU/Serial++.h \
		TU/TU/Array++.h \
		TU/TU/Geometry++.h \
		TU/TU/List++.h \
		TU/TU/TU/Vector++.h \
		TU/TU/TU/types.h \
		TU/TU/functions.h \
		TU/Vector++.cc \
		TU/mmInstructions.h
HDRS		= Allocator++.h \
		Array++.h \
		Bezier++.h \
		BlockMatrix++.h \
		Geometry++.h \
		Heap++.h \
		Image++.h \
		List++.h \
		Manip.h \
		Mesh++.h \
		Minimize++.h \
		Nurbs++.h \
		PSTree++.h \
		Random.h \
		Serial++.h \
		Vector++.h \
		functions.h \
		mmInstructions.h \
		types.h
SRCS		= Allocator++.cc \
		Array++.cc \
		Array++.inst.cc \
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
		EdgeDetector.cc \
		Geometry++.cc \
		Geometry++.inst.cc \
		Heap++.cc \
		IIRFilter.cc \
		Image++.cc \
		Image++.inst.cc \
		ImageBase.cc \
		ImageLine.cc \
		List++.cc \
		Mesh++.cc \
		Microscope.cc \
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
		manipulators.cc
OBJS		= Allocator++.o \
		Array++.o \
		Array++.inst.o \
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
		EdgeDetector.o \
		Geometry++.o \
		Geometry++.inst.o \
		Heap++.o \
		IIRFilter.o \
		Image++.o \
		Image++.inst.o \
		ImageBase.o \
		ImageLine.o \
		List++.o \
		Mesh++.o \
		Microscope.o \
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
		manipulators.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.32 $	|		\
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
Array++.o: TU/TU/Array++.h TU/TU/TU/types.h
Array++.inst.o: TU/Array++.cc TU/TU/Array++.h TU/TU/TU/types.h
Bezier++.o: TU/Bezier++.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h
BlockMatrix++.o: TU/BlockMatrix++.h TU/TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
BlockMatrix++.inst.o: TU/BlockMatrix++.cc TU/BlockMatrix++.h \
	TU/TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h TU/Array++.cc
Camera.o: TU/TU/Geometry++.h TU/TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
CameraBase.o: TU/TU/Geometry++.h TU/TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
CameraWithDistortion.o: TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h
CameraWithEuclideanImagePlane.o: TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h
CameraWithFocalLength.o: TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h
CanonicalCamera.o: TU/TU/Geometry++.h TU/TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
ConversionFromYUV.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h
EdgeDetector.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h TU/mmInstructions.h
Geometry++.o: TU/TU/Geometry++.h TU/TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h
Geometry++.inst.o: TU/TU/Geometry++.h TU/TU/TU/Vector++.h TU/TU/Array++.h \
	TU/TU/TU/types.h TU/Geometry++.cc
Heap++.o: TU/Heap++.h TU/TU/Array++.h TU/TU/TU/types.h
IIRFilter.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h TU/Minimize++.h TU/Array++.cc \
	TU/Image++.cc TU/TU/functions.h TU/mmInstructions.h
Image++.o: TU/TU/functions.h TU/Image++.h TU/TU/Geometry++.h \
	TU/TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/mmInstructions.h
Image++.inst.o: TU/Array++.cc TU/TU/Array++.h TU/TU/TU/types.h \
	TU/Image++.cc TU/TU/functions.h TU/Image++.h TU/TU/Geometry++.h \
	TU/TU/TU/Vector++.h TU/mmInstructions.h
ImageBase.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h TU/Manip.h
ImageLine.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h
List++.o: TU/TU/List++.h
Mesh++.o: TU/Mesh++.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h TU/Allocator++.h TU/TU/List++.h
Microscope.o: TU/Serial++.h TU/Manip.h TU/TU/Geometry++.h \
	TU/TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h TU/Geometry++.cc
Nurbs++.o: TU/TU/functions.h TU/Nurbs++.h TU/TU/Geometry++.h \
	TU/TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
PSTree++.o: TU/PSTree++.h TU/Heap++.h TU/TU/Array++.h TU/TU/TU/types.h \
	TU/TU/List++.h
Pata.o: TU/Serial++.h TU/Manip.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h TU/Geometry++.cc
Puma.o: TU/Serial++.h TU/Manip.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h TU/Geometry++.cc
Random.o: TU/Random.h
Rotation.o: TU/TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
Serial.o: TU/Serial++.h TU/Manip.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h TU/Geometry++.cc
TUTools++.sa.o: TU/Image++.h TU/TU/Geometry++.h TU/TU/TU/Vector++.h \
	TU/TU/Array++.h TU/TU/TU/types.h TU/Serial++.h TU/Manip.h \
	TU/Geometry++.cc
TriggerGenerator.o: TU/Serial++.h TU/Manip.h TU/TU/Geometry++.h \
	TU/TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h TU/Geometry++.cc
Vector++.o: TU/TU/TU/Vector++.h TU/TU/Array++.h TU/TU/TU/types.h
Vector++.inst.o: TU/Array++.cc TU/TU/Array++.h TU/TU/TU/types.h \
	TU/Vector++.cc TU/TU/TU/Vector++.h
