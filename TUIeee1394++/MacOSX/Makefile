#
#  $Id: Makefile,v 1.5 2009-10-16 02:47:25 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= /usr/local/lib
INCDIR		= /usr/local/include/libraw1394
INCDIRS		= -I.

NAME		= raw1394

CPPFLAGS	= #-DSINGLE_THREAD #-DDEBUG
CFLAGS		= -g
CCFLAGS		= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  CCFLAGS	= $(CFLAGS)
endif

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		=
HDRS		= csr.h \
		raw1394.h \
		raw1394_.h
SRCS		= raw1394.cc
OBJS		= raw1394.o

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
###
raw1394.o: raw1394_.h raw1394.h
