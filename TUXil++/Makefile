#
#  $Id: Makefile,v 1.5 2002-08-28 01:37:29 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU/v
INCDIRS		= -I. -I$(HOME)/include -I$(OPENWINHOME)/include -I$(XILHOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DUseXaw -DDEBUG #-DTUXilPP_UsePlaneMasks
CFLAGS		= -O
CCFLAGS		= -O

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		=
HDRS		= XilDC.h
SRCS		= TUXil++.inst.cc \
		TUXil++.sa.cc \
		XilDC.cc \
		XilImage.cc \
		XilObject.cc
OBJS		= TUXil++.inst.o \
		TUXil++.sa.o \
		XilDC.o \
		XilImage.o \
		XilObject.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.5 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
include $(PROJECT)/lib/RCS.mk
###
