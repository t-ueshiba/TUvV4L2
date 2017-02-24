#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
PROGRAM		= array_test
#LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I..
CPPFLAGS	= -DTU_DEBUG #-DCPP11
CFLAGS		= -std=c++1y -g
NVCCFLAGS	= -O
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		=
LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../TU/Array++.h \
		../TU/algorithm.h \
		../TU/iterator.h \
		../TU/range.h \
		../TU/tuple.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Common macros	#
#########################
ifneq ($(findstring darwin,$(OSTYPE)),)
  ifneq ($(strip $(LIBRARY)),)
    LIBRARY    := $(LIBRARY).dylib
  endif
  PIC		= -dynamic
  NVCCFLAGS    += -m64
  ifneq ($(findstring nvcc,$(LINKER)),)
    SHFLAGS	= $(NVCCFLAGS) -shared -Xlinker "-dynamic -undefined dynamic_lookup"
  else
    SHFLAGS	= -dynamiclib -undefined dynamic_lookup 
  endif
else
  ifneq ($(strip $(LIBRARY)),)
    LIBRARY    := $(LIBRARY).so
  endif
  PIC		= -fPIC
  SHFLAGS	= -shared
endif

ifneq ($(strip $(PROGRAM)),)
  TARGPROG	= $(BINDIR)/$(PROGRAM)
endif
ifneq ($(strip $(LIBRARY)),)
  TARGLIB	= $(LIBDIR)/$(LIBRARY)
endif

INSTALL		= install
MAKEFILE	= Makefile

ifeq ($(shell arch), armv7l)
  CFLAGS       += -mcpu=cortex-a7 -mtune=cortex-a7 -mfpu=neon -funsafe-math-optimizations #-mfloat-abi=softfp
else ifeq ($(shell arch), aarch64)
  CFLAGS       += -funsafe-math-optimizations
endif

ifneq ($(findstring icpc,$(CXX)),)
    LIBS       += -lsvml -lintlc
endif

#########################
#  Making rules		#
#########################
.PHONY: all
all: $(PROGRAM) $(LIBRARY)

$(SRCS): $(IDLHDRS)

$(PROGRAM): $(OBJS)
	$(LINKER) $(LDFLAGS) $^ -o $@ $(LIBS)

$(LIBRARY): $(OBJS)
	$(LINKER) $(SHFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

$(TARGPROG): $(PROGRAM)
	$(INSTALL) -m 0755 $< $@

$(TARGLIB): $(LIBRARY)
	$(INSTALL) -m 0755 $< $@

.PHONY: install
install: $(TARGPROG) $(TARGLIB) $(TARGHDRS)

.PHONY:	clean
clean:
	$(RM) -r $(IDLSRCS) $(IDLHDRS) $(MOCSRCS) $(OBJS) $(PROGRAM) $(LIBRARY)

.PHONY: depend
depend:
	mkmf $(INCDIRS) -f $(MAKEFILE)

.PHONY: index
index:
	ctags -wx $(HDRS) $(SRCS)

tags: $(HDRS) $(SRCS)
	ctags $(HDRS) $(SRCS)

.PHONY: link
link:
	$(RM) $(PROGRAM) $(LIBRARY)
	make $(PROGRAM) $(LIBRARY)

.PHONY:	doc
doc: $(HDRS) $(SRCS) doxygen.cfg
	doxygen doxygen.cfg

doxygen.cfg:
	doxygen -g $@

#########################
#  Implicit rules	#
#########################
$(INCDIR)/%: %
	$(INSTALL) -m 0644 $< $@

%.hh %SK.cc %DynSK.cc: %.idl
	$(IDLC) $(IDLFLAGS) $<

moc_%.cc: %.h
	$(MOC) $< -o $@

%.o: %.c
	$(CC) $(CPPFLAGS) $(INCDIRS) $(CFLAGS) $(PIC) -c $<

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(INCDIRS) $(CCFLAGS) $(PIC) -c $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(INCDIRS) $(CCFLAGS) $(PIC) -c $<

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(INCDIRS) $(NVCCFLAGS) -Xcompiler $(PIC) -c $<
###
main.o: ../TU/Array++.h ../TU/range.h ../TU/algorithm.h ../TU/tuple.h \
	../TU/iterator.h
