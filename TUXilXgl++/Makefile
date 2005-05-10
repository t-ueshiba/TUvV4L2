#
#  $Id: Makefile,v 1.7 2005-05-10 02:51:16 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU/v
INCDIRS		= -I$(HOME)/include -I$(OPENWINHOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -DUseXaw #-DUseShm
CFLAGS		= -O
CCFLAGS		= -O

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= TU/v/XilXglDC.h
HDRS		= XilXglDC.h
SRCS		= XilXglDC.cc
OBJS		= XilXglDC.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.7 $	|		\
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
XilXglDC.o: TU/v/XilXglDC.h
