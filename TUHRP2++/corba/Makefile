#########################################
#  Target and IDL sources		#
#########################################
PREFIX		= /usr/local
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
NAME		= HRP2Corba

IDLSRCS		= ReachingService.idl \
		  SequencePlayerService.idl \
		  WalkGeneratorService.idl
IDLSRCS2	= ForwardKinematicsService.idl

#########################################
#  IDL compiler				#
#########################################
IDL		= omniidl
IDLFLAGS	= -bcxx -Wbh=.h -Wbs=.cpp -Wba -Wbd=DynSK.cpp -Wbtf
IDLFLAGS2	= -bcxx -Wbh=.hh -Wbs=.cc -Wba -Wbd=DynSK.cc -Wbtf

#########################################
#  C++ preprocessor and compiler	#
#########################################
INCDIRS		= -I. -I/usr/include/rtm/idl -I$(PREFIX)/include/rtm/idl -I/usr/local/include/rtm/idl
CPPFLAGS	= -DOMNIORB4 -DTVMET_OPTIMIZE -DNDEBUG -DUSE_CLAPACK_INTERFACE
CCC		= g++
CCFLAGS		= -fPIC -O3 -Wall -fno-schedule-insns -fno-schedule-insns2 -fno-strict-aliasing
LINKER		= $(CCC)

#################################
#  Intermediate files		#
#################################
.SUFFIXES:	.cc .cpp .idl

HDRS		= $(patsubst %.idl,%.h,$(IDLSRCS))
SRCS		= $(patsubst %.idl,%.cpp,$(IDLSRCS)) \
		  $(patsubst %.idl,%DynSK.cpp,$(IDLSRCS))
OBJS		= $(patsubst %.cpp,%.o,$(SRCS))
HDRS2		= $(patsubst %.idl,%.hh,$(IDLSRCS2))
SRCS2		= $(patsubst %.idl,%.cc,$(IDLSRCS2)) \
		  $(patsubst %.idl,%DynSK.cc,$(IDLSRCS2))
OBJS2		= $(patsubst %.cc,%.o,$(SRCS2))

SOLIB		= lib$(NAME).so
DESTCOMMHDRS	= $(HDRS:%=$(INCDIR)/%) $(HDRS2:%=$(INCDIR)/%)
DESTLIBRARY	= $(SOLIB:%=$(DEST)/%)

INSTALL		= install
PRINT		= pr
MAKEFILE	= Makefile

#################################
#  Making rules			#
#################################
all:		$(SOLIB)

install:	$(DESTLIBRARY) $(DESTCOMMHDRS)

$(DEST)/$(SOLIB):	$(SOLIB)
		$(INSTALL) -m 0755 $(SOLIB) $@

clean:
		$(RM) $(SOLIB) $(OBJS) $(SRCS) $(HDRS) $(OBJS2) $(SRCS2) $(HDRS2)

$(SOLIB):	$(OBJS) $(OBJS2)
		$(LINKER) -shared $(LDFLAGS) $(OBJS) $(OBJS2) -o $@

$(OBJS):	$(SRCS) $(HDRS)

$(OBJS2):	$(SRCS2) $(HDRS2)

$(SRCS) $(HDRS):$(IDLSRCS)

$(SRCS2) $(HDRS2):$(IDLSRCS2)

$(INCDIR)/%:	%
		$(INSTALL) -m 0644 $< $(INCDIR)

.idl.cpp:
		$(IDL) $(IDLFLAGS) $(INCDIRS) $<

.cpp.o:
		$(CCC) $(CPPFLAGS) $(CCFLAGS) $(INCDIRS) -c $<

.idl.cc:
		$(IDL) $(IDLFLAGS2) $(INCDIRS) $<

.cc.o:
		$(CCC) $(CPPFLAGS) $(CCFLAGS) $(INCDIRS) -c $<
