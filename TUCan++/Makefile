#
#  $Id: Makefile,v 1.4 2007-09-30 23:30:09 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
LIBDIR		= $(HOME)/lib
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I$(HOME)/include -I/usr/local/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g
CCFLAGS		= -g

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /Users/ueshiba/include/TU/Array++.h \
		/Users/ueshiba/include/TU/Geometry++.cc \
		/Users/ueshiba/include/TU/Geometry++.h \
		/Users/ueshiba/include/TU/Minimize++.h \
		/Users/ueshiba/include/TU/Vector++.h \
		/Users/ueshiba/include/TU/types.h \
		/Users/ueshiba/include/TU/utility.h \
		TU/Can++.h
HDRS		= Can++.h
SRCS		= Can.cc \
		Manus.cc
OBJS		= Can.o \
		Manus.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.4 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

#########################
#  Common macros	#
#########################
COMMHDRS	= $(filter-out %_.h, $(HDRS)) $(filter %++.cc, $(SRCS))
DESTCOMMHDRS	= $(COMMHDRS:%=$(INCDIR)/%)

LIBOBJS		= $(filter-out %++.o, $(OBJS))

VER		= $(shell echo $(REV) | awk -F"." '{printf("%d", $$1);}')

ALIB		= lib$(NAME).a
SLINK		= lib$(NAME).dylib
SOLIB		= lib$(NAME).$(VER).dylib

ifneq ($(strip $(LIBOBJS)),)
#   LIBRARY    += $(ALIB)
    LIBRARY    += $(SOLIB)
endif
DESTLIBRARY	= $(LIBRARY:%=$(DEST)/%)

CPIC	= -dynamic
CCPIC	= -dynamic

LN		= ln -s
INSTALL		= install -c
PRINT		= pr
MAKEFILE	= Makefile

#########################
#  Making rules		#
#########################
all:		archive $(LIBRARY)

$(ALIB):	$(LIBOBJS)
		$(RM) $@
#		$(LINKER) -xar -o $@ archive/$(LIBOBJS)
		(cd archive; $(AR) rv ../$@ $(LIBOBJS))
		ranlib $@

$(SOLIB):	$(LIBOBJS)
		$(LINKER) -dynamiclib -undefined dynamic_lookup -o $@ $(LIBOBJS)
		@$(RM) $(SLINK)
		@$(LN) $(SOLIB) $(SLINK)

archive:
		mkdir archive

install:	$(DESTLIBRARY) $(DESTCOMMHDRS)

$(DEST)/$(ALIB):	$(ALIB)
		$(INSTALL) -m 0644 $(ALIB) $@
		ranlib $@

$(DEST)/$(SOLIB):	$(SOLIB)
		$(INSTALL) -m 0755 $(SOLIB) $@
		@$(RM) $(DEST)/$(SLINK)
		@$(LN) $(SOLIB) $(DEST)/$(SLINK)

clean:
		$(RM) -r $(LIBRARY) $(SLINK) $(OBJS) .sb
		(cd archive; $(RM) $(OBJS))

depend:
		mkmf $(INCDIRS) -f $(MAKEFILE)

index:
		ctags -wx $(HDRS) $(SRCS)

tags:		$(HDRS) $(SRCS)
		ctags $(HDRS) $(SRCS)

print:
		$(PRINT) $(HDRS) $(SRCS)

doc:		$(HDRS) $(SRCS) doxygen.conf
		doxygen doxygen.conf

doxygen.conf:
		doxygen -g $@

#########################
#  Implicit rules	#
#########################
$(INCDIR)/%:	%
		$(INSTALL) -m 0644 $< $(INCDIR)

.c.o:
		$(CC) $(CPPFLAGS) $(CFLAGS) $(CPIC) $(INCDIRS) -c $<
#		$(CC) $(CPPFLAGS) $(CFLAGS) $(INCDIRS) -c $< -o archive/$@

.cc.o:
		$(CCC) $(CPPFLAGS) $(CCFLAGS) $(CCPIC) $(INCDIRS) -c $<
#		$(CCC) $(CPPFLAGS) $(CCFLAGS) $(INCDIRS) -c $< -o archive/$@
###
Can.o: TU/Can++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/utility.h \
	/Users/ueshiba/include/TU/Minimize++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h
Manus.o: TU/Can++.h /Users/ueshiba/include/TU/Geometry++.h \
	/Users/ueshiba/include/TU/utility.h \
	/Users/ueshiba/include/TU/Minimize++.h \
	/Users/ueshiba/include/TU/Vector++.h \
	/Users/ueshiba/include/TU/Array++.h /Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Geometry++.cc
