#
#  $Id: Makefile,v 1.16 2007-01-31 05:42:44 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -O -g
CCFLAGS		= -O -g
ifeq ($(CCC), icpc)
  CCFLAGS	= -O3 -parallel
endif
LDFLAGS		= $(CCFLAGS)
LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Array++.cc \
		/Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/BlockMatrix++.h \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Minimize++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/types.h \
		TU/Calib++.cc \
		TU/Calib++.h
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
REV		= $(shell echo $Revision: 1.16 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Calib++.o: TU/Calib++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
MeasurementMatrix.o: TU/Calib++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h \
	/Users/ueshiba/include/TU/Array++.cc TU/Calib++.cc
Normalization.o: TU/Calib++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
affine_registration.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
assess_projection.o: TU/Calib++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
decompose_essential.o: TU/Calib++.h /Users/ueshiba/include/TU/Geometry++.h \
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
io.o: TU/Calib++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
plane_motion.o: TU/Calib++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
projection.o: TU/Calib++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/BlockMatrix++.h \
	/Users/ueshiba/include/TU/Minimize++.h
similarity_registration.o: /Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
