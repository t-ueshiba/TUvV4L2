#
#  $Id: Makefile,v 1.100 2012-08-30 01:58:16 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I../stereo1394 -I$(PREFIX)/include

PROGRAM		= stereoV4L2

LIBS		= -lTUV4L2++ -lTUOgl++ -lTUv++ -lTUStereo++ -lTUTools++ \
		  -lGLU -lGL \
		  -lXaw3d -lXt -lXext -lXmu -lX11 -lraw1394 -ltbb -ltbbmalloc

CPPFLAGS	= -DNDEBUG -DHAVE_LIBTUTOOLS__ -DUSE_TBB \
		  -DDISPLAY_2D -DDISPLAY_3D #-DUSE_VIDEO1394 #-DNO_VERTICAL_BM \
		  -DNO_RV #-DDIRECT_CAPTURE
CFLAGS		= -g
NVCCFLAGS	= -g
ifneq ($(findstring icpc,$(CXX)),)
  CFLAGS	= -O3 -xAVX
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DAVX
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= ../stereo1394/ComputeThreeD.h \
		../stereo1394/DrawThreeD.h \
		../stereo1394/MyCanvasPane.h \
		../stereo1394/MyOglCanvasPane.h \
		../stereo1394/MyOglWindow.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/BoxFilter.h \
		/usr/local/include/TU/Camera++.h \
		/usr/local/include/TU/GFStereo.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/Rectify.h \
		/usr/local/include/TU/SADStereo.h \
		/usr/local/include/TU/SeparableFilter2.h \
		/usr/local/include/TU/StereoBase.h \
		/usr/local/include/TU/V4L2++.h \
		/usr/local/include/TU/V4L2CameraArray.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/Warp.h \
		/usr/local/include/TU/algorithm.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/io.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/v/App.h \
		/usr/local/include/TU/v/CanvasPane.h \
		/usr/local/include/TU/v/CanvasPaneDC.h \
		/usr/local/include/TU/v/CanvasPaneDC3.h \
		/usr/local/include/TU/v/CmdPane.h \
		/usr/local/include/TU/v/CmdWindow.h \
		/usr/local/include/TU/v/Colormap.h \
		/usr/local/include/TU/v/DC.h \
		/usr/local/include/TU/v/DC3.h \
		/usr/local/include/TU/v/Dialog.h \
		/usr/local/include/TU/v/FileSelection.h \
		/usr/local/include/TU/v/Menu.h \
		/usr/local/include/TU/v/ModalDialog.h \
		/usr/local/include/TU/v/Notify.h \
		/usr/local/include/TU/v/OglDC.h \
		/usr/local/include/TU/v/ShmDC.h \
		/usr/local/include/TU/v/TUv++.h \
		/usr/local/include/TU/v/Timer.h \
		/usr/local/include/TU/v/Widget-Xaw.h \
		/usr/local/include/TU/v/XDC.h
HDRS		= MyCmdWindow.h \
		stereoV4L2.h
SRCS		= ComputeThreeD.cc \
		DrawThreeD.cc \
		MyCanvasPane.cc \
		MyCmdWindow.cc \
		MyOglCanvasPane.cc \
		createCaptureCmds.cc \
		createFeatureCmds.cc \
		createMenuCmds.cc \
		main.cc
OBJS		= ComputeThreeD.o \
		DrawThreeD.o \
		MyCanvasPane.o \
		MyCmdWindow.o \
		MyOglCanvasPane.o \
		createCaptureCmds.o \
		createFeatureCmds.o \
		createMenuCmds.o \
		main.o

include $(PROJECT)/lib/p.mk

###
ComputeThreeD.o: ../stereo1394/ComputeThreeD.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h
DrawThreeD.o: ../stereo1394/DrawThreeD.h /usr/local/include/TU/Warp.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Camera++.h
MyCanvasPane.o: ../stereo1394/MyCanvasPane.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/ShmDC.h
MyCmdWindow.o: /usr/local/include/TU/v/FileSelection.h \
	/usr/local/include/TU/v/ModalDialog.h \
	/usr/local/include/TU/v/Dialog.h /usr/local/include/TU/v/CmdPane.h \
	/usr/local/include/TU/v/CmdWindow.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h /usr/local/include/TU/v/Notify.h \
	stereoV4L2.h /usr/local/include/TU/V4L2++.h MyCmdWindow.h \
	/usr/local/include/TU/v/App.h /usr/local/include/TU/v/Timer.h \
	/usr/local/include/TU/V4L2CameraArray.h /usr/local/include/TU/io.h \
	/usr/local/include/TU/Rectify.h /usr/local/include/TU/Warp.h \
	/usr/local/include/TU/Camera++.h ../stereo1394/MyCanvasPane.h \
	/usr/local/include/TU/v/CanvasPane.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/ShmDC.h \
	../stereo1394/MyOglWindow.h ../stereo1394/MyOglCanvasPane.h \
	/usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h /usr/local/include/TU/v/DC3.h \
	../stereo1394/DrawThreeD.h ../stereo1394/ComputeThreeD.h \
	/usr/local/include/TU/SADStereo.h /usr/local/include/TU/StereoBase.h \
	/usr/local/include/TU/algorithm.h /usr/local/include/TU/Profiler.h \
	/usr/local/include/TU/BoxFilter.h \
	/usr/local/include/TU/SeparableFilter2.h \
	/usr/local/include/TU/GFStereo.h
MyOglCanvasPane.o: ../stereo1394/MyOglCanvasPane.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h /usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/DC3.h \
	../stereo1394/DrawThreeD.h /usr/local/include/TU/Warp.h \
	/usr/local/include/TU/Camera++.h
createCaptureCmds.o: stereoV4L2.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h /usr/local/include/TU/V4L2++.h
createFeatureCmds.o: stereoV4L2.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h /usr/local/include/TU/V4L2++.h
createMenuCmds.o: stereoV4L2.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h /usr/local/include/TU/V4L2++.h
main.o: /usr/local/include/TU/SADStereo.h \
	/usr/local/include/TU/StereoBase.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/algorithm.h \
	/usr/local/include/TU/Profiler.h /usr/local/include/TU/BoxFilter.h \
	/usr/local/include/TU/SeparableFilter2.h \
	/usr/local/include/TU/GFStereo.h MyCmdWindow.h \
	/usr/local/include/TU/v/App.h /usr/local/include/TU/v/TUv++.h \
	/usr/local/include/TU/v/Colormap.h /usr/local/include/TU/List.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/CmdWindow.h /usr/local/include/TU/v/CmdPane.h \
	/usr/local/include/TU/v/Timer.h \
	/usr/local/include/TU/V4L2CameraArray.h \
	/usr/local/include/TU/V4L2++.h /usr/local/include/TU/io.h \
	/usr/local/include/TU/Rectify.h /usr/local/include/TU/Warp.h \
	/usr/local/include/TU/Camera++.h ../stereo1394/MyCanvasPane.h \
	/usr/local/include/TU/v/CanvasPane.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/ShmDC.h \
	../stereo1394/MyOglWindow.h ../stereo1394/MyOglCanvasPane.h \
	/usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h /usr/local/include/TU/v/DC3.h \
	../stereo1394/DrawThreeD.h
