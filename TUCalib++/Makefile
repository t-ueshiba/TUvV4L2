#
#  $Id: Makefile,v 1.26 2009-07-08 01:10:18 ueshiba Exp $
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
ifeq ($(CCC), icpc)
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS	= -O3 -axP
  else
    CPPFLAGS   += -DSSSE3
    CFLAGS	= -O3 -xN
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/BlockMatrix++.h \
		/usr/local/include/TU/Camera.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Normalize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/types.h \
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
REV		= $(shell echo $Revision: 1.26 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Calib++.o: TU/Calib++.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Minimize.h
MeasurementMatrix.o: TU/Calib++.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Minimize.h TU/Calib++.cc
Normalization.o: TU/Calib++.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Minimize.h
affine_registration.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
assess_projection.o: TU/Calib++.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Minimize.h
decompose_essential.o: TU/Calib++.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Minimize.h
decompose_homography.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
euclidean_registration.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
get_flengths.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
get_principal.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
io.o: TU/Calib++.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Minimize.h
plane_motion.o: TU/Calib++.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Minimize.h
projection.o: TU/Calib++.h /usr/local/include/TU/Camera.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h \
	/usr/local/include/TU/BlockMatrix++.h \
	/usr/local/include/TU/Minimize.h
similarity_registration.o: /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
