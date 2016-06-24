#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
PROGRAM		= $(shell basename $(PWD))
#LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

CC		= gcc
CXX		= g++
INCDIRS		= -I.. -I$(PREFIX)/include \
		  -I$(PREFIX)/Make/libraw1394-2.1.2/src \
		  -I$(PREFIX)/Make/libraw1394-2.1.2
CPPFLAGS	= -DNDEBUG -DHAVE_LIBTUTOOLS__ #-DDEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUTools++ -lusb-1.0
ifneq ($(findstring darwin,$(OSTYPE)),)
  LIBS	       += -framework CoreServices
endif

LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../FireWireNode_.h \
		../TU/IIDC++.h \
		../USBNode_.h \
		/usr/local/Make/libraw1394-2.1.2/config.h \
		/usr/local/Make/libraw1394-2.1.2/src/csr.h \
		/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
		/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
		/usr/local/Make/libraw1394-2.1.2/src/fw.h \
		/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
		/usr/local/Make/libraw1394-2.1.2/src/ieee1394.h \
		/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
		/usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
		/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h
HDRS		=
SRCS		= FireWireNode.cc \
		IIDCCamera.cc \
		IIDCNode.cc \
		USBNode.cc \
		arm.c \
		dispatch.c \
		errors.c \
		eventloop.c \
		fcp.c \
		fw-iso.c \
		fw.c \
		iso.c \
		libraw1394-main.c \
		main.cc \
		readwrite.c \
		version.c
OBJS		= FireWireNode.o \
		IIDCCamera.o \
		IIDCNode.o \
		USBNode.o \
		arm.o \
		dispatch.o \
		errors.o \
		eventloop.o \
		fcp.o \
		fw-iso.o \
		fw.o \
		iso.o \
		libraw1394-main.o \
		main.o \
		readwrite.o \
		version.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
FireWireNode.o: ../FireWireNode_.h ../TU/IIDC++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/Minimize.h
IIDCCamera.o: ../FireWireNode_.h ../TU/IIDC++.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/Minimize.h \
	../USBNode_.h
IIDCNode.o: ../TU/IIDC++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Minimize.h
USBNode.o: ../USBNode_.h ../TU/IIDC++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Minimize.h
arm.o: /usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/config.h
dispatch.o: /usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/config.h
errors.o: /usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/config.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394.h
eventloop.o: /usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/config.h
fcp.o: /usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/config.h
fw-iso.o: /usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/config.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h
fw.o: /usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/config.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h
iso.o: /usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/config.h
libraw1394-main.o: /usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/config.h
main.o: ../TU/IIDC++.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Minimize.h
readwrite.o: /usr/local/Make/libraw1394-2.1.2/src/raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/kernel-raw1394.h \
	/usr/local/Make/libraw1394-2.1.2/src/ieee1394-ioctl.h \
	/usr/local/Make/libraw1394-2.1.2/src/raw1394_private.h \
	/usr/local/Make/libraw1394-2.1.2/src/csr.h \
	/usr/local/Make/libraw1394-2.1.2/src/fw.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-constants.h \
	/usr/local/Make/libraw1394-2.1.2/src/firewire-cdev.h \
	/usr/local/Make/libraw1394-2.1.2/config.h
