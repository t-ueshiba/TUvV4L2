#
#  $Id: Makefile,v 1.3 2002-08-12 01:24:46 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include -I$(SNPHOME)/lib/src/include -I$(OPENWINHOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -D_SOLARIS2 -D_REENTRANT -D_SNP24 -D_X_GRAPHICS -DUseXaw
CFLAGS		= -g
CCFLAGS		= -g

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		=
HDRS		= Snapper24++.h \
		TUSnapper24++.h
SRCS		= Snapper24++.cc \
		Snapper24.cc \
		snp24.c
OBJS		= Snapper24++.o \
		Snapper24.o \
		snp24.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.3 $	|		\
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
