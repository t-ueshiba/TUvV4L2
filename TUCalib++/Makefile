#
#  $Id: Makefile,v 1.11 2002-12-18 06:04:32 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include
SRCDIR		= $(HOME)/VVV/Pkg/vvv/libraries/libTUCalib++

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DTUMinimizePP_DEBUG
CFLAGS		= -g -O
CCFLAGS		= -g -O

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Array++.cc \
		/home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/BlockMatrix++.h \
		/home/ueshiba/include/TU/Geometry++.h \
		/home/ueshiba/include/TU/Minimize++.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/types.h \
		TU/Calib++.h
HDRS		= Calib++.h
SRCS		= MeasurementMatrix.cc \
		Normalization.cc \
		affine_registration.cc \
		assess_projection.cc \
		decompose_essential.cc \
		decompose_homography.cc \
		euclidean_registration.cc \
		get_flengths.cc \
		get_principal.cc \
		io.cc \
		plane_motion.cc \
		projection.cc \
		similarity_registration.cc
OBJS		= MeasurementMatrix.o \
		Normalization.o \
		affine_registration.o \
		assess_projection.o \
		decompose_essential.o \
		decompose_homography.o \
		euclidean_registration.o \
		get_flengths.o \
		get_principal.o \
		io.o \
		plane_motion.o \
		projection.o \
		similarity_registration.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.11 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
include $(PROJECT)/lib/l-install-srcs.mk
###
MeasurementMatrix.o: TU/Calib++.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/BlockMatrix++.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Array++.cc
Normalization.o: TU/Calib++.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/BlockMatrix++.h
affine_registration.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
assess_projection.o: TU/Calib++.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/BlockMatrix++.h
decompose_essential.o: TU/Calib++.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/BlockMatrix++.h
decompose_homography.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
euclidean_registration.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
get_flengths.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
get_principal.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
io.o: TU/Calib++.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/BlockMatrix++.h
plane_motion.o: TU/Calib++.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/BlockMatrix++.h
projection.o: TU/Calib++.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/BlockMatrix++.h
similarity_registration.o: /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
