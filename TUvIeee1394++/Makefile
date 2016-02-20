#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
#PROGRAM		= $(shell basename $(PWD))
LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I. -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG -DHAVE_LIBTUTOOLS__ #-DUSE_VIDEO1394
CFLAGS		= -O3
NVCCFLAGS	= -O
ifeq ($(shell arch), armv7l)
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
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Ieee1394++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/v/CmdPane.h \
		/usr/local/include/TU/v/CmdWindow.h \
		/usr/local/include/TU/v/Colormap.h \
		/usr/local/include/TU/v/Dialog.h \
		/usr/local/include/TU/v/ModalDialog.h \
		/usr/local/include/TU/v/TUv++.h \
		/usr/local/include/TU/v/Widget-Xaw.h
HDRS		= TU/v/vIeee1394++.h
SRCS		= createFeatureCmds.cc \
		createFormatMenu.cc \
		setSpecialFormat.cc
OBJS		= createFeatureCmds.o \
		createFormatMenu.o \
		setSpecialFormat.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
createFeatureCmds.o: TU/v/vIeee1394++.h /usr/local/include/TU/v/CmdPane.h \
	/usr/local/include/TU/v/CmdWindow.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/Ieee1394++.h
createFormatMenu.o: TU/v/vIeee1394++.h /usr/local/include/TU/v/CmdPane.h \
	/usr/local/include/TU/v/CmdWindow.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/Ieee1394++.h
setSpecialFormat.o: TU/v/vIeee1394++.h /usr/local/include/TU/v/CmdPane.h \
	/usr/local/include/TU/v/CmdWindow.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/Ieee1394++.h \
	/usr/local/include/TU/v/ModalDialog.h \
	/usr/local/include/TU/v/Dialog.h
