#
#  $Id: Makefile,v 1.13 2004-06-17 00:28:04 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include
SRCDIR		= $(HOME)/VVV/Pkg/vvv/libraries/libTUCalib++

NAME		= $(shell basename $(PWD))

CPPFLAGS	= #-DTUMinimizePP_DEBUG
CFLAGS		= -g -O
CCFLAGS		= -g -O

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Array++.cc \
		/Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/BlockMatrix++.h \
		/Users/ueshiba/include/TU/Calib++.cc \
		/Users/ueshiba/include/TU/Calib++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Minimize++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/types.h
HDRS		= Calib++.h
SRCS		= Calib++.cc \
		MeasurementMatrix.cc \
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
OBJS		= Calib++.o \
		MeasurementMatrix.o \
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
REV		= $(shell echo $Revision: 1.13 $	|		\
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
Calib++.o: /Users/ueshiba/include/TU/Calib++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
MeasurementMatrix.o: /Users/ueshiba/include/TU/Calib++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h \
	/Users/ueshiba/include/TU/Array++.cc \
	/Users/ueshiba/include/TU/Calib++.cc
Normalization.o: /Users/ueshiba/include/TU/Calib++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
affine_registration.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
assess_projection.o: /Users/ueshiba/include/TU/Calib++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
decompose_essential.o: /Users/ueshiba/include/TU/Calib++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
decompose_homography.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
euclidean_registration.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
get_flengths.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
get_principal.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
io.o: /Users/ueshiba/include/TU/Calib++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
plane_motion.o: /Users/ueshiba/include/TU/Calib++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
projection.o: /Users/ueshiba/include/TU/Calib++.h \
	/Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
similarity_registration.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
