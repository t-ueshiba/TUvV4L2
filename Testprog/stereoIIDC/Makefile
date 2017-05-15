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

INCDIRS		= -I../.. -I../stereoGUI -I$(PREFIX)/include
CPPFLAGS	= -DNDEBUG -DUSE_TBB -DDISPLAY_2D -DDISPLAY_3D \
		#-DNO_RV -DUSE_VIDEO1394 -DDIRECT_CAPTURE
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

LIBS		= -lTUvIIDC++ -lTUIIDC++ -lTUOgl++ -lTUv++ -lTUTools++ \
		  -lGLU -lGL -lXaw3d -lXt -lXext -lXmu -lX11 \
		  -lraw1394 -lusb-1.0 -ltbb -ltbbmalloc
ifeq ($(shell arch), aarch64)
  LIBS	       += -ldrm
endif

LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../../TU/Array++.h \
		../../TU/BoxFilter.h \
		../../TU/Camera++.h \
		../../TU/Filter2.h \
		../../TU/GFStereo.h \
		../../TU/Geometry++.h \
		../../TU/Image++.h \
		../../TU/List.h \
		../../TU/Manip.h \
		../../TU/Minimize.h \
		../../TU/Profiler.h \
		../../TU/Rectify.h \
		../../TU/SADStereo.h \
		../../TU/StereoBase.h \
		../../TU/Vector++.h \
		../../TU/Warp.h \
		../../TU/algorithm.h \
		../../TU/functional.h \
		../../TU/io.h \
		../../TU/iterator.h \
		../../TU/pair.h \
		../../TU/range.h \
		../../TU/simd/BufTraits.h \
		../../TU/simd/allocator.h \
		../../TU/simd/arithmetic.h \
		../../TU/simd/arm/allocator.h \
		../../TU/simd/arm/arch.h \
		../../TU/simd/arm/arithmetic.h \
		../../TU/simd/arm/bit_shift.h \
		../../TU/simd/arm/cast.h \
		../../TU/simd/arm/compare.h \
		../../TU/simd/arm/cvt.h \
		../../TU/simd/arm/dup.h \
		../../TU/simd/arm/insert_extract.h \
		../../TU/simd/arm/load_store.h \
		../../TU/simd/arm/logical.h \
		../../TU/simd/arm/lookup.h \
		../../TU/simd/arm/select.h \
		../../TU/simd/arm/shift.h \
		../../TU/simd/arm/type_traits.h \
		../../TU/simd/arm/vec.h \
		../../TU/simd/arm/zero.h \
		../../TU/simd/bit_shift.h \
		../../TU/simd/cast.h \
		../../TU/simd/compare.h \
		../../TU/simd/config.h \
		../../TU/simd/cvt.h \
		../../TU/simd/cvtdown_iterator.h \
		../../TU/simd/cvtup_iterator.h \
		../../TU/simd/dup.h \
		../../TU/simd/insert_extract.h \
		../../TU/simd/load_iterator.h \
		../../TU/simd/load_store.h \
		../../TU/simd/logical.h \
		../../TU/simd/lookup.h \
		../../TU/simd/pack.h \
		../../TU/simd/select.h \
		../../TU/simd/shift.h \
		../../TU/simd/shift_iterator.h \
		../../TU/simd/simd.h \
		../../TU/simd/store_iterator.h \
		../../TU/simd/transform.h \
		../../TU/simd/type_traits.h \
		../../TU/simd/vec.h \
		../../TU/simd/x86/allocator.h \
		../../TU/simd/x86/arch.h \
		../../TU/simd/x86/arithmetic.h \
		../../TU/simd/x86/bit_shift.h \
		../../TU/simd/x86/cast.h \
		../../TU/simd/x86/compare.h \
		../../TU/simd/x86/cvt.h \
		../../TU/simd/x86/dup.h \
		../../TU/simd/x86/insert_extract.h \
		../../TU/simd/x86/load_store.h \
		../../TU/simd/x86/logical.h \
		../../TU/simd/x86/logical_base.h \
		../../TU/simd/x86/lookup.h \
		../../TU/simd/x86/select.h \
		../../TU/simd/x86/shift.h \
		../../TU/simd/x86/type_traits.h \
		../../TU/simd/x86/unpack.h \
		../../TU/simd/x86/vec.h \
		../../TU/simd/x86/zero.h \
		../../TU/simd/zero.h \
		../../TU/tuple.h \
		../stereoGUI/ComputeThreeD.h \
		../stereoGUI/DrawThreeD.h \
		../stereoGUI/MyCanvasPane.h \
		../stereoGUI/MyOglCanvasPane.h \
		../stereoGUI/stereoGUI.h \
		/usr/local/include/TU/IIDC++.h \
		/usr/local/include/TU/IIDCCameraArray.h \
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
		/usr/local/include/TU/v/XDC.h \
		/usr/local/include/TU/v/vIIDC++.h
HDRS		= MyCmdWindow.h \
		MyOglWindow.h
SRCS		= ComputeThreeD.cc \
		DrawThreeD.cc \
		MyCanvasPane.cc \
		MyOglCanvasPane.cc \
		main.cc
OBJS		= ComputeThreeD.o \
		DrawThreeD.o \
		MyCanvasPane.o \
		MyOglCanvasPane.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
ComputeThreeD.o: ../stereoGUI/ComputeThreeD.h ../../TU/Geometry++.h \
	../../TU/Minimize.h ../../TU/Vector++.h ../../TU/Array++.h \
	../../TU/range.h ../../TU/algorithm.h ../../TU/iterator.h \
	../../TU/tuple.h
DrawThreeD.o: ../stereoGUI/DrawThreeD.h ../../TU/Warp.h ../../TU/Image++.h \
	../../TU/pair.h ../../TU/Vector++.h ../../TU/Array++.h \
	../../TU/range.h ../../TU/algorithm.h ../../TU/iterator.h \
	../../TU/tuple.h ../../TU/Camera++.h ../../TU/Geometry++.h \
	../../TU/Minimize.h ../../TU/simd/simd.h ../../TU/simd/config.h \
	../../TU/simd/vec.h ../../TU/simd/type_traits.h \
	../../TU/simd/x86/type_traits.h ../../TU/simd/arm/type_traits.h \
	../../TU/simd/x86/vec.h ../../TU/simd/x86/arch.h \
	../../TU/simd/arm/vec.h ../../TU/simd/arm/arch.h \
	../../TU/simd/allocator.h ../../TU/simd/x86/allocator.h \
	../../TU/simd/arm/allocator.h ../../TU/simd/load_store.h \
	../../TU/simd/pack.h ../../TU/simd/cvt.h ../../TU/simd/zero.h \
	../../TU/simd/x86/zero.h ../../TU/simd/arm/zero.h \
	../../TU/simd/cast.h ../../TU/simd/x86/cast.h \
	../../TU/simd/arm/cast.h ../../TU/simd/shift.h \
	../../TU/simd/x86/shift.h ../../TU/simd/arm/shift.h \
	../../TU/simd/bit_shift.h ../../TU/simd/x86/bit_shift.h \
	../../TU/simd/arm/bit_shift.h ../../TU/simd/x86/cvt.h \
	../../TU/simd/x86/unpack.h ../../TU/simd/arm/cvt.h \
	../../TU/simd/x86/load_store.h ../../TU/simd/arm/load_store.h \
	../../TU/simd/insert_extract.h ../../TU/simd/x86/insert_extract.h \
	../../TU/simd/arm/insert_extract.h ../../TU/simd/dup.h \
	../../TU/simd/logical.h ../../TU/simd/x86/logical.h \
	../../TU/simd/x86/logical_base.h ../../TU/simd/arm/logical.h \
	../../TU/simd/x86/dup.h ../../TU/simd/arm/dup.h \
	../../TU/simd/compare.h ../../TU/simd/x86/compare.h \
	../../TU/simd/arm/compare.h ../../TU/simd/select.h \
	../../TU/simd/x86/select.h ../../TU/simd/arm/select.h \
	../../TU/simd/arithmetic.h ../../TU/simd/x86/arithmetic.h \
	../../TU/simd/arm/arithmetic.h ../../TU/simd/transform.h \
	../../TU/functional.h ../../TU/simd/lookup.h \
	../../TU/simd/x86/lookup.h ../../TU/simd/arm/lookup.h \
	../../TU/simd/load_iterator.h ../../TU/simd/store_iterator.h \
	../../TU/simd/cvtdown_iterator.h ../../TU/simd/cvtup_iterator.h \
	../../TU/simd/shift_iterator.h ../../TU/simd/BufTraits.h
MyCanvasPane.o: ../stereoGUI/MyCanvasPane.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	../../TU/List.h ../../TU/Geometry++.h ../../TU/Minimize.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/range.h \
	../../TU/algorithm.h ../../TU/iterator.h ../../TU/tuple.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h ../../TU/Image++.h ../../TU/pair.h \
	../../TU/Manip.h /usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/ShmDC.h
MyOglCanvasPane.o: ../stereoGUI/MyOglCanvasPane.h \
	/usr/local/include/TU/v/CanvasPane.h /usr/local/include/TU/v/TUv++.h \
	../../TU/List.h ../../TU/Geometry++.h ../../TU/Minimize.h \
	../../TU/Vector++.h ../../TU/Array++.h ../../TU/range.h \
	../../TU/algorithm.h ../../TU/iterator.h ../../TU/tuple.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h ../../TU/Image++.h ../../TU/pair.h \
	../../TU/Manip.h /usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/DC3.h \
	../stereoGUI/DrawThreeD.h ../../TU/Warp.h ../../TU/Camera++.h \
	../../TU/simd/simd.h ../../TU/simd/config.h ../../TU/simd/vec.h \
	../../TU/simd/type_traits.h ../../TU/simd/x86/type_traits.h \
	../../TU/simd/arm/type_traits.h ../../TU/simd/x86/vec.h \
	../../TU/simd/x86/arch.h ../../TU/simd/arm/vec.h \
	../../TU/simd/arm/arch.h ../../TU/simd/allocator.h \
	../../TU/simd/x86/allocator.h ../../TU/simd/arm/allocator.h \
	../../TU/simd/load_store.h ../../TU/simd/pack.h ../../TU/simd/cvt.h \
	../../TU/simd/zero.h ../../TU/simd/x86/zero.h \
	../../TU/simd/arm/zero.h ../../TU/simd/cast.h \
	../../TU/simd/x86/cast.h ../../TU/simd/arm/cast.h \
	../../TU/simd/shift.h ../../TU/simd/x86/shift.h \
	../../TU/simd/arm/shift.h ../../TU/simd/bit_shift.h \
	../../TU/simd/x86/bit_shift.h ../../TU/simd/arm/bit_shift.h \
	../../TU/simd/x86/cvt.h ../../TU/simd/x86/unpack.h \
	../../TU/simd/arm/cvt.h ../../TU/simd/x86/load_store.h \
	../../TU/simd/arm/load_store.h ../../TU/simd/insert_extract.h \
	../../TU/simd/x86/insert_extract.h ../../TU/simd/arm/insert_extract.h \
	../../TU/simd/dup.h ../../TU/simd/logical.h \
	../../TU/simd/x86/logical.h ../../TU/simd/x86/logical_base.h \
	../../TU/simd/arm/logical.h ../../TU/simd/x86/dup.h \
	../../TU/simd/arm/dup.h ../../TU/simd/compare.h \
	../../TU/simd/x86/compare.h ../../TU/simd/arm/compare.h \
	../../TU/simd/select.h ../../TU/simd/x86/select.h \
	../../TU/simd/arm/select.h ../../TU/simd/arithmetic.h \
	../../TU/simd/x86/arithmetic.h ../../TU/simd/arm/arithmetic.h \
	../../TU/simd/transform.h ../../TU/functional.h \
	../../TU/simd/lookup.h ../../TU/simd/x86/lookup.h \
	../../TU/simd/arm/lookup.h ../../TU/simd/load_iterator.h \
	../../TU/simd/store_iterator.h ../../TU/simd/cvtdown_iterator.h \
	../../TU/simd/cvtup_iterator.h ../../TU/simd/shift_iterator.h \
	../../TU/simd/BufTraits.h
main.o: ../../TU/SADStereo.h ../../TU/StereoBase.h ../../TU/Image++.h \
	../../TU/pair.h ../../TU/Vector++.h ../../TU/Array++.h \
	../../TU/range.h ../../TU/algorithm.h ../../TU/iterator.h \
	../../TU/tuple.h ../../TU/simd/simd.h ../../TU/simd/config.h \
	../../TU/simd/vec.h ../../TU/simd/type_traits.h \
	../../TU/simd/x86/type_traits.h ../../TU/simd/arm/type_traits.h \
	../../TU/simd/x86/vec.h ../../TU/simd/x86/arch.h \
	../../TU/simd/arm/vec.h ../../TU/simd/arm/arch.h \
	../../TU/simd/allocator.h ../../TU/simd/x86/allocator.h \
	../../TU/simd/arm/allocator.h ../../TU/simd/load_store.h \
	../../TU/simd/pack.h ../../TU/simd/cvt.h ../../TU/simd/zero.h \
	../../TU/simd/x86/zero.h ../../TU/simd/arm/zero.h \
	../../TU/simd/cast.h ../../TU/simd/x86/cast.h \
	../../TU/simd/arm/cast.h ../../TU/simd/shift.h \
	../../TU/simd/x86/shift.h ../../TU/simd/arm/shift.h \
	../../TU/simd/bit_shift.h ../../TU/simd/x86/bit_shift.h \
	../../TU/simd/arm/bit_shift.h ../../TU/simd/x86/cvt.h \
	../../TU/simd/x86/unpack.h ../../TU/simd/arm/cvt.h \
	../../TU/simd/x86/load_store.h ../../TU/simd/arm/load_store.h \
	../../TU/simd/insert_extract.h ../../TU/simd/x86/insert_extract.h \
	../../TU/simd/arm/insert_extract.h ../../TU/simd/dup.h \
	../../TU/simd/logical.h ../../TU/simd/x86/logical.h \
	../../TU/simd/x86/logical_base.h ../../TU/simd/arm/logical.h \
	../../TU/simd/x86/dup.h ../../TU/simd/arm/dup.h \
	../../TU/simd/compare.h ../../TU/simd/x86/compare.h \
	../../TU/simd/arm/compare.h ../../TU/simd/select.h \
	../../TU/simd/x86/select.h ../../TU/simd/arm/select.h \
	../../TU/simd/arithmetic.h ../../TU/simd/x86/arithmetic.h \
	../../TU/simd/arm/arithmetic.h ../../TU/simd/transform.h \
	../../TU/functional.h ../../TU/simd/lookup.h \
	../../TU/simd/x86/lookup.h ../../TU/simd/arm/lookup.h \
	../../TU/simd/load_iterator.h ../../TU/simd/store_iterator.h \
	../../TU/simd/cvtdown_iterator.h ../../TU/simd/cvtup_iterator.h \
	../../TU/simd/shift_iterator.h ../../TU/simd/BufTraits.h \
	../../TU/Profiler.h ../../TU/BoxFilter.h ../../TU/Filter2.h \
	../../TU/GFStereo.h ../../TU/io.h /usr/local/include/TU/v/vIIDC++.h \
	/usr/local/include/TU/v/CmdPane.h /usr/local/include/TU/v/CmdWindow.h \
	/usr/local/include/TU/v/TUv++.h ../../TU/List.h ../../TU/Geometry++.h \
	../../TU/Minimize.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/v/Widget-Xaw.h \
	/usr/local/include/TU/v/Colormap.h \
	/usr/local/include/TU/IIDCCameraArray.h \
	/usr/local/include/TU/IIDC++.h MyCmdWindow.h \
	/usr/local/include/TU/v/App.h /usr/local/include/TU/v/Timer.h \
	/usr/local/include/TU/v/FileSelection.h \
	/usr/local/include/TU/v/ModalDialog.h \
	/usr/local/include/TU/v/Dialog.h /usr/local/include/TU/v/Notify.h \
	../../TU/Rectify.h ../../TU/Warp.h ../../TU/Camera++.h \
	../stereoGUI/MyCanvasPane.h /usr/local/include/TU/v/CanvasPane.h \
	/usr/local/include/TU/v/CanvasPaneDC.h /usr/local/include/TU/v/XDC.h \
	/usr/local/include/TU/v/DC.h ../../TU/Manip.h \
	/usr/local/include/TU/v/Menu.h /usr/local/include/TU/v/ShmDC.h \
	../stereoGUI/ComputeThreeD.h MyOglWindow.h \
	../stereoGUI/MyOglCanvasPane.h /usr/local/include/TU/v/OglDC.h \
	/usr/local/include/TU/v/CanvasPaneDC3.h /usr/local/include/TU/v/DC3.h \
	../stereoGUI/DrawThreeD.h ../stereoGUI/stereoGUI.h
