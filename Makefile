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
CPPFLAGS	= -DNDEBUG
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
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/List.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/simd/allocator.h \
		/usr/local/include/TU/simd/arithmetic.h \
		/usr/local/include/TU/simd/arm/allocator.h \
		/usr/local/include/TU/simd/arm/arch.h \
		/usr/local/include/TU/simd/arm/arithmetic.h \
		/usr/local/include/TU/simd/arm/bit_shift.h \
		/usr/local/include/TU/simd/arm/cast.h \
		/usr/local/include/TU/simd/arm/compare.h \
		/usr/local/include/TU/simd/arm/cvt.h \
		/usr/local/include/TU/simd/arm/cvt_mask.h \
		/usr/local/include/TU/simd/arm/insert_extract.h \
		/usr/local/include/TU/simd/arm/load_store.h \
		/usr/local/include/TU/simd/arm/logical.h \
		/usr/local/include/TU/simd/arm/select.h \
		/usr/local/include/TU/simd/arm/shift.h \
		/usr/local/include/TU/simd/arm/type_traits.h \
		/usr/local/include/TU/simd/arm/vec.h \
		/usr/local/include/TU/simd/arm/zero.h \
		/usr/local/include/TU/simd/bit_shift.h \
		/usr/local/include/TU/simd/cast.h \
		/usr/local/include/TU/simd/compare.h \
		/usr/local/include/TU/simd/config.h \
		/usr/local/include/TU/simd/cvt.h \
		/usr/local/include/TU/simd/cvt_mask.h \
		/usr/local/include/TU/simd/cvtdown_iterator.h \
		/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
		/usr/local/include/TU/simd/cvtup_iterator.h \
		/usr/local/include/TU/simd/cvtup_mask_iterator.h \
		/usr/local/include/TU/simd/insert_extract.h \
		/usr/local/include/TU/simd/intel/allocator.h \
		/usr/local/include/TU/simd/intel/arch.h \
		/usr/local/include/TU/simd/intel/arithmetic.h \
		/usr/local/include/TU/simd/intel/bit_shift.h \
		/usr/local/include/TU/simd/intel/cast.h \
		/usr/local/include/TU/simd/intel/compare.h \
		/usr/local/include/TU/simd/intel/cvt.h \
		/usr/local/include/TU/simd/intel/cvt_mask.h \
		/usr/local/include/TU/simd/intel/dup.h \
		/usr/local/include/TU/simd/intel/insert_extract.h \
		/usr/local/include/TU/simd/intel/load_store.h \
		/usr/local/include/TU/simd/intel/logical.h \
		/usr/local/include/TU/simd/intel/logical_base.h \
		/usr/local/include/TU/simd/intel/select.h \
		/usr/local/include/TU/simd/intel/shift.h \
		/usr/local/include/TU/simd/intel/shuffle.h \
		/usr/local/include/TU/simd/intel/svml.h \
		/usr/local/include/TU/simd/intel/type_traits.h \
		/usr/local/include/TU/simd/intel/unpack.h \
		/usr/local/include/TU/simd/intel/vec.h \
		/usr/local/include/TU/simd/intel/zero.h \
		/usr/local/include/TU/simd/load_iterator.h \
		/usr/local/include/TU/simd/load_store.h \
		/usr/local/include/TU/simd/logical.h \
		/usr/local/include/TU/simd/misc.h \
		/usr/local/include/TU/simd/row_vec_iterator.h \
		/usr/local/include/TU/simd/select.h \
		/usr/local/include/TU/simd/shift.h \
		/usr/local/include/TU/simd/shift_iterator.h \
		/usr/local/include/TU/simd/simd.h \
		/usr/local/include/TU/simd/store_iterator.h \
		/usr/local/include/TU/simd/type_traits.h \
		/usr/local/include/TU/simd/vec.h \
		/usr/local/include/TU/simd/zero.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h
HDRS		= ButtonCmd_.h \
		ChoiceFrameCmd_.h \
		ChoiceMenuButtonCmd_.h \
		FrameCmd_.h \
		LabelCmd_.h \
		ListCmd_.h \
		MenuButtonCmd_.h \
		RadioButtonCmd_.h \
		SliderCmd_.h \
		TU/v/App.h \
		TU/v/Bitmap.h \
		TU/v/CanvasPane.h \
		TU/v/CanvasPaneDC.h \
		TU/v/CanvasPaneDC3.h \
		TU/v/CmdPane.h \
		TU/v/CmdWindow.h \
		TU/v/Colormap.h \
		TU/v/Confirm.h \
		TU/v/DC.h \
		TU/v/DC3.h \
		TU/v/Dialog.h \
		TU/v/FileSelection.h \
		TU/v/Icon.h \
		TU/v/MemoryDC.h \
		TU/v/Menu.h \
		TU/v/ModalDialog.h \
		TU/v/Notify.h \
		TU/v/ShmDC.h \
		TU/v/TUv++.h \
		TU/v/Timer.h \
		TU/v/Widget-Xaw.h \
		TU/v/XDC.h \
		TextInCmd_.h \
		ToggleButtonCmd_.h \
		vCanvasP_.h \
		vCanvas_.h \
		vGridboxP_.h \
		vGridbox_.h \
		vSliderP_.h \
		vSlider_.h \
		vTextFieldP_.h \
		vTextField_.h \
		vViewportP_.h \
		vViewport_.h
SRCS		= App.cc \
		Bitmap.cc \
		ButtonCmd.cc \
		CanvasPane.cc \
		CanvasPaneDC.cc \
		CanvasPaneDC3.cc \
		ChoiceFrameCmd.cc \
		ChoiceMenuButtonCmd.cc \
		Cmd.cc \
		CmdPane.cc \
		CmdParent.cc \
		CmdWindow.cc \
		Colormap.cc \
		Confirm.cc \
		DC.cc \
		DC3.cc \
		Dialog.cc \
		FileSelection.cc \
		FrameCmd.cc \
		Icon.cc \
		LabelCmd.cc \
		ListCmd.cc \
		MemoryDC.cc \
		Menu.cc \
		MenuButtonCmd.cc \
		ModalDialog.cc \
		Notify.cc \
		Object.cc \
		Pane.cc \
		RadioButtonCmd.cc \
		ShmDC.cc \
		SliderCmd.cc \
		TUv++.inst.cc \
		TextInCmd.cc \
		Timer.cc \
		ToggleButtonCmd.cc \
		Widget-Xaw.cc \
		Window.cc \
		XDC.cc \
		vCanvas.c \
		vGridbox.c \
		vSlider.c \
		vTextField.c \
		vViewport.c
OBJS		= App.o \
		Bitmap.o \
		ButtonCmd.o \
		CanvasPane.o \
		CanvasPaneDC.o \
		CanvasPaneDC3.o \
		ChoiceFrameCmd.o \
		ChoiceMenuButtonCmd.o \
		Cmd.o \
		CmdPane.o \
		CmdParent.o \
		CmdWindow.o \
		Colormap.o \
		Confirm.o \
		DC.o \
		DC3.o \
		Dialog.o \
		FileSelection.o \
		FrameCmd.o \
		Icon.o \
		LabelCmd.o \
		ListCmd.o \
		MemoryDC.o \
		Menu.o \
		MenuButtonCmd.o \
		ModalDialog.o \
		Notify.o \
		Object.o \
		Pane.o \
		RadioButtonCmd.o \
		ShmDC.o \
		SliderCmd.o \
		TUv++.inst.o \
		TextInCmd.o \
		Timer.o \
		ToggleButtonCmd.o \
		Widget-Xaw.o \
		Window.o \
		XDC.o \
		vCanvas.o \
		vGridbox.o \
		vSlider.o \
		vTextField.o \
		vViewport.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
App.o: TU/v/App.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
Bitmap.o: TU/v/Bitmap.h TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h
ButtonCmd.o: ButtonCmd_.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h TU/v/Bitmap.h
CanvasPane.o: TU/v/CanvasPane.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h vViewport_.h vGridbox_.h
CanvasPaneDC.o: TU/v/CanvasPaneDC.h TU/v/XDC.h TU/v/DC.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Manip.h \
	TU/v/Colormap.h TU/v/CanvasPane.h TU/v/TUv++.h \
	/usr/local/include/TU/List.h TU/v/Widget-Xaw.h TU/v/Menu.h vCanvas_.h \
	vViewport_.h
CanvasPaneDC3.o: TU/v/CanvasPaneDC3.h TU/v/CanvasPaneDC.h TU/v/XDC.h \
	TU/v/DC.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Manip.h \
	TU/v/Colormap.h TU/v/CanvasPane.h TU/v/TUv++.h \
	/usr/local/include/TU/List.h TU/v/Widget-Xaw.h TU/v/Menu.h TU/v/DC3.h
ChoiceFrameCmd.o: ChoiceFrameCmd_.h FrameCmd_.h TU/v/TUv++.h \
	TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
ChoiceMenuButtonCmd.o: ChoiceMenuButtonCmd_.h TU/v/Menu.h TU/v/TUv++.h \
	TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
Cmd.o: TU/v/TUv++.h TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h LabelCmd_.h SliderCmd_.h FrameCmd_.h ButtonCmd_.h \
	TU/v/Bitmap.h ToggleButtonCmd_.h MenuButtonCmd_.h TU/v/Menu.h \
	ChoiceMenuButtonCmd_.h RadioButtonCmd_.h ChoiceFrameCmd_.h ListCmd_.h \
	TextInCmd_.h
CmdPane.o: TU/v/CmdPane.h TU/v/CmdWindow.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h vGridbox_.h
CmdParent.o: TU/v/TUv++.h TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
CmdWindow.o: TU/v/CmdWindow.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h TU/v/App.h vGridbox_.h
Colormap.o: TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h
Confirm.o: TU/v/Confirm.h TU/v/ModalDialog.h TU/v/Dialog.h TU/v/CmdPane.h \
	TU/v/CmdWindow.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
DC.o: TU/v/DC.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Manip.h
DC3.o: TU/v/DC3.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/types.h
Dialog.o: TU/v/Dialog.h TU/v/CmdPane.h TU/v/CmdWindow.h TU/v/TUv++.h \
	TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
FileSelection.o: TU/v/FileSelection.h TU/v/ModalDialog.h TU/v/Dialog.h \
	TU/v/CmdPane.h TU/v/CmdWindow.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h TU/v/Notify.h TU/v/Confirm.h
FrameCmd.o: FrameCmd_.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
Icon.o: TU/v/Icon.h TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h
LabelCmd.o: LabelCmd_.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
ListCmd.o: ListCmd_.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h vViewport_.h
MemoryDC.o: TU/v/MemoryDC.h TU/v/XDC.h TU/v/DC.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Manip.h \
	TU/v/Colormap.h TU/v/CanvasPane.h TU/v/TUv++.h \
	/usr/local/include/TU/List.h TU/v/Widget-Xaw.h
Menu.o: TU/v/Menu.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h TU/v/Bitmap.h
MenuButtonCmd.o: MenuButtonCmd_.h TU/v/Menu.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
ModalDialog.o: TU/v/ModalDialog.h TU/v/Dialog.h TU/v/CmdPane.h \
	TU/v/CmdWindow.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
Notify.o: TU/v/Notify.h TU/v/ModalDialog.h TU/v/Dialog.h TU/v/CmdPane.h \
	TU/v/CmdWindow.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
Object.o: TU/v/TUv++.h TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
Pane.o: TU/v/TUv++.h TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h vGridbox_.h
RadioButtonCmd.o: TU/v/Bitmap.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h RadioButtonCmd_.h TU/v/TUv++.h \
	/usr/local/include/TU/List.h TU/v/Widget-Xaw.h vGridbox_.h
ShmDC.o: TU/v/ShmDC.h TU/v/CanvasPaneDC.h TU/v/XDC.h TU/v/DC.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Manip.h \
	TU/v/Colormap.h TU/v/CanvasPane.h TU/v/TUv++.h \
	/usr/local/include/TU/List.h TU/v/Widget-Xaw.h TU/v/Menu.h
SliderCmd.o: SliderCmd_.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h vSlider_.h vGridbox_.h
TUv++.inst.o: TU/v/TUv++.h TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
TextInCmd.o: TextInCmd_.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h vTextField_.h
Timer.o: TU/v/Timer.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h TU/v/App.h
ToggleButtonCmd.o: ToggleButtonCmd_.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h TU/v/Bitmap.h
Widget-Xaw.o: TU/v/TUv++.h TU/v/Colormap.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h vGridbox_.h vTextField_.h vViewport_.h
Window.o: TU/v/App.h TU/v/TUv++.h TU/v/Colormap.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/List.h \
	TU/v/Widget-Xaw.h
XDC.o: TU/v/XDC.h TU/v/DC.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/intel/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/intel/vec.h \
	/usr/local/include/TU/simd/intel/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/intel/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/intel/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/intel/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/intel/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/intel/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/intel/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/intel/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/intel/cvt.h \
	/usr/local/include/TU/simd/intel/dup.h \
	/usr/local/include/TU/simd/intel/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/cvt_mask.h \
	/usr/local/include/TU/simd/intel/cvt_mask.h \
	/usr/local/include/TU/simd/arm/cvt_mask.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/intel/logical.h \
	/usr/local/include/TU/simd/intel/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/intel/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/intel/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/intel/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/intel/shuffle.h \
	/usr/local/include/TU/simd/intel/svml.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/cvtdown_mask_iterator.h \
	/usr/local/include/TU/simd/cvtup_mask_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Manip.h \
	TU/v/Colormap.h
vCanvas.o: vCanvasP_.h vCanvas_.h
vGridbox.o: vGridboxP_.h vGridbox_.h
vSlider.o: vSliderP_.h vSlider_.h
vTextField.o: vTextFieldP_.h vTextField_.h
vViewport.o: vViewportP_.h vViewport_.h
