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

INCDIRS		= -I../.. -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG -DTU_MESH_DEBUG
CFLAGS		= -O3
NVCCFLAGS	= -O
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else ifeq ($(shell arch), aarch64)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE4
  CFLAGS       += -msse4
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUOgl++ -lTUv++ -lTUTools++ \
		  -lGLU -lGL -lXaw3d -lXt -lXext -lXmu -lX11
LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC .idl:hC
EXTHDRS		= ../../TU/Array++.h \
		../../TU/Camera++.h \
		../../TU/Geometry++.h \
		../../TU/Image++.h \
		../../TU/List.h \
		../../TU/Manip.h \
		../../TU/Mesh++.h \
		../../TU/Minimize.h \
		../../TU/Vector++.h \
		../../TU/algorithm.h \
		../../TU/iterator.h \
		../../TU/pair.h \
		../../TU/range.h \
		../../TU/tuple.h \
		../../TU/type_traits.h \
		/usr/local/include/TU/cuda/iterator.h \
		/usr/local/include/TU/cuda/tuple.h \
		/usr/local/include/TU/v/App.h \
		/usr/local/include/TU/v/CanvasPane.h \
		/usr/local/include/TU/v/CanvasPaneDC.h \
		/usr/local/include/TU/v/CanvasPaneDC3.h \
		/usr/local/include/TU/v/CmdPane.h \
		/usr/local/include/TU/v/CmdWindow.h \
		/usr/local/include/TU/v/Colormap.h \
		/usr/local/include/TU/v/DC.h \
		/usr/local/include/TU/v/DC3.h \
		/usr/local/include/TU/v/Menu.h \
		/usr/local/include/TU/v/OglDC.h \
		/usr/local/include/TU/v/TUv++.h \
		/usr/local/include/TU/v/Widget-Xaw.h \
		/usr/local/include/TU/v/XDC.h
HDRS		= TriMesh.h
SRCS		= draw.cc \
		main.cc
OBJS		= draw.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
draw.o: TriMesh.h ../../TU/Mesh++.h ../../TU/Geometry++.h \
	../../TU/Minimize.h ../../TU/Vector++.h ../../TU/Array++.h \
	../../TU/range.h ../../TU/iterator.h ../../TU/tuple.h \
	../../TU/type_traits.h ../../TU/algorithm.h \
	/usr/local/include/TU/cuda/tuple.h \
	/usr/local/include/TU/cuda/iterator.h ../../TU/Manip.h \
	/usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h ../../TU/Image++.h ../../TU/pair.h \
	../../TU/Camera++.h /usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	../../TU/List.h /usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/DC3.h
main.o: /usr/local/include/TU/v/App.h /usr/local/include/TU/v/TUv++.h \
	../../TU/List.h ../../TU/Geometry++.h ../../TU/Minimize.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/range.h \
	../../TU/iterator.h ../../TU/tuple.h ../../TU/type_traits.h \
	../../TU/algorithm.h /usr/local/include/TU/cuda/tuple.h \
	/usr/local/include/TU/cuda/iterator.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/Colormap.h ../../TU/Image++.h ../../TU/pair.h \
	../../TU/Manip.h ../../TU/Camera++.h \
	/usr/local/include/TU/v/CmdWindow.h /usr/local/include/TU/v/CmdPane.h \
	/usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/v/CanvasPane.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/DC3.h \
	TriMesh.h ../../TU/Mesh++.h
