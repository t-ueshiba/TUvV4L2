#
#  $Id: Makefile,v 1.4 2012-09-01 05:37:02 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I../testcam -I$(HOME)/multicam/server -I$(PREFIX)/include

PROGRAM		= $(shell basename $(PWD))
LIBS		= -lTUvIeee1394++ -lTUIeee1394++ -lTUXv++ -lTUv++ -lTUTools++ \
		  -lXaw3d -lXv -lXt -lXext -lXmu -lX11 -lraw1394
ifeq ($(OSTYPE), darwin)
  LIBS	       += -framework IOKit -framework CoreFoundation
endif

CPPFLAGS	= -DHAVE_LIBTUTOOLS__
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= ../testcam/MyCanvasPane.h \
		../testcam/testcam.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Ieee1394++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Movie.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/v/App.h \
		/usr/local/include/TU/v/CanvasPane.h \
		/usr/local/include/TU/v/CanvasPaneDC.h \
		/usr/local/include/TU/v/CmdPane.h \
		/usr/local/include/TU/v/CmdWindow.h \
		/usr/local/include/TU/v/Colormap.h \
		/usr/local/include/TU/v/DC.h \
		/usr/local/include/TU/v/Dialog.h \
		/usr/local/include/TU/v/FileSelection.h \
		/usr/local/include/TU/v/Menu.h \
		/usr/local/include/TU/v/ModalDialog.h \
		/usr/local/include/TU/v/ShmDC.h \
		/usr/local/include/TU/v/TUv++.h \
		/usr/local/include/TU/v/Timer.h \
		/usr/local/include/TU/v/Widget-Xaw.h \
		/usr/local/include/TU/v/XDC.h \
		/usr/local/include/TU/v/XvDC.h \
		/usr/local/include/TU/v/vIeee1394++.h
HDRS		= MyCmdWindow.h
SRCS		= main.cc
OBJS		= main.o

include $(PROJECT)/lib/p.mk

###
main.o: MyCmdWindow.h /usr/local/include/TU/v/App.h \
	/usr/local/include/TU/v/TUv++.h /usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/List.h /usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/CmdWindow.h /usr/local/include/TU/v/CmdPane.h \
	/usr/local/include/TU/v/FileSelection.h \
	/usr/local/include/TU/v/ModalDialog.h \
	/usr/local/include/TU/v/Dialog.h /usr/local/include/TU/v/Timer.h \
	/usr/local/include/TU/v/vIeee1394++.h \
	/usr/local/include/TU/Ieee1394++.h /usr/local/include/TU/Movie.h \
	/usr/local/include/TU/Manip.h ../testcam/testcam.h \
	../testcam/MyCanvasPane.h /usr/local/include/TU/v/CanvasPane.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/v/Menu.h \
	/usr/local/include/TU/v/ShmDC.h /usr/local/include/TU/v/XvDC.h
