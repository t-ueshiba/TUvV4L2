#
#  $Id: Makefile,v 1.19 2008-06-09 00:10:43 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU/v
INCDIRS		= -I. -I$(HOME)/include -I$(X11HOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g
ifeq ($(CCC), icpc)
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS	= -O3 -axP -parallel -ip
  else
    CPPFLAGS   += -DSSE2
    CFLAGS	= -O3 -tpp7 -xW -ip
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Array++.h \
		/home/ueshiba/include/TU/Geometry++.h \
		/home/ueshiba/include/TU/Image++.h \
		/home/ueshiba/include/TU/List++.cc \
		/home/ueshiba/include/TU/List++.h \
		/home/ueshiba/include/TU/Manip.h \
		/home/ueshiba/include/TU/Minimize++.h \
		/home/ueshiba/include/TU/Vector++.h \
		/home/ueshiba/include/TU/types.h \
		/home/ueshiba/include/TU/utility.h \
		TU/v/App.h \
		TU/v/Bitmap.h \
		TU/v/CanvasPane.h \
		TU/v/CanvasPaneDC.h \
		TU/v/CanvasPaneDC3.h \
		TU/v/CmdPane.h \
		TU/v/Confirm.h \
		TU/v/FileSelection.h \
		TU/v/Icon.h \
		TU/v/MemoryDC.h \
		TU/v/Notify.h \
		TU/v/ShmDC.h \
		TU/v/TU/v/CmdWindow.h \
		TU/v/TU/v/DC3.h \
		TU/v/TU/v/Menu.h \
		TU/v/TU/v/ModalDialog.h \
		TU/v/TU/v/TU/v/Colormap.h \
		TU/v/TU/v/TU/v/DC.h \
		TU/v/TU/v/TU/v/Dialog.h \
		TU/v/TU/v/TU/v/Widget-Xaw.h \
		TU/v/TU/v/TUv++.h \
		TU/v/TU/v/XDC.h \
		TU/v/Timer.h
HDRS		= App.h \
		Bitmap.h \
		ButtonCmd_.h \
		CanvasPane.h \
		CanvasPaneDC.h \
		CanvasPaneDC3.h \
		ChoiceFrameCmd_.h \
		ChoiceMenuButtonCmd_.h \
		CmdPane.h \
		CmdWindow.h \
		Colormap.h \
		Confirm.h \
		DC.h \
		DC3.h \
		Dialog.h \
		FileSelection.h \
		FrameCmd_.h \
		Icon.h \
		LabelCmd_.h \
		ListCmd_.h \
		MemoryDC.h \
		Menu.h \
		MenuButtonCmd_.h \
		ModalDialog.h \
		Notify.h \
		RadioButtonCmd_.h \
		ShmDC.h \
		SliderCmd_.h \
		TUv++.h \
		TextInCmd_.h \
		Timer.h \
		ToggleButtonCmd_.h \
		Widget-Xaw.h \
		XDC.h \
		vCanvasP_.h \
		vCanvas_.h \
		vGridboxP_.h \
		vGridbox_.h \
		vScrollbarP_.h \
		vScrollbar_.h \
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
		vScrollbar.c \
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
		vScrollbar.o \
		vSlider.o \
		vTextField.o \
		vViewport.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.19 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
App.o: TU/v/App.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
Bitmap.o: TU/v/Bitmap.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
ButtonCmd.o: ButtonCmd_.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/Bitmap.h
CanvasPane.o: TU/v/CanvasPane.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	vViewport_.h vGridbox_.h
CanvasPaneDC.o: TU/v/CanvasPaneDC.h TU/v/TU/v/XDC.h TU/v/TU/v/TU/v/DC.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h \
	TU/v/TU/v/TU/v/Colormap.h TU/v/CanvasPane.h TU/v/TU/v/TUv++.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/TU/v/Menu.h vCanvas_.h vViewport_.h
CanvasPaneDC3.o: TU/v/CanvasPaneDC3.h TU/v/CanvasPaneDC.h TU/v/TU/v/XDC.h \
	TU/v/TU/v/TU/v/DC.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h \
	TU/v/TU/v/TU/v/Colormap.h TU/v/CanvasPane.h TU/v/TU/v/TUv++.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/TU/v/Menu.h TU/v/TU/v/DC3.h
ChoiceFrameCmd.o: ChoiceFrameCmd_.h FrameCmd_.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
ChoiceMenuButtonCmd.o: ChoiceMenuButtonCmd_.h TU/v/TU/v/Menu.h \
	TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
Cmd.o: TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	LabelCmd_.h SliderCmd_.h FrameCmd_.h ButtonCmd_.h TU/v/Bitmap.h \
	ToggleButtonCmd_.h MenuButtonCmd_.h TU/v/TU/v/Menu.h \
	ChoiceMenuButtonCmd_.h RadioButtonCmd_.h ChoiceFrameCmd_.h ListCmd_.h \
	TextInCmd_.h
CmdPane.o: TU/v/CmdPane.h TU/v/TU/v/CmdWindow.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	vGridbox_.h
CmdParent.o: TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
CmdWindow.o: TU/v/TU/v/CmdWindow.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/App.h vGridbox_.h
Colormap.o: TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
Confirm.o: TU/v/Confirm.h TU/v/TU/v/ModalDialog.h TU/v/TU/v/TU/v/Dialog.h \
	TU/v/CmdPane.h TU/v/TU/v/CmdWindow.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
DC.o: TU/v/TU/v/TU/v/DC.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h
DC3.o: TU/v/TU/v/DC3.h /home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Manip.h
Dialog.o: TU/v/TU/v/TU/v/Dialog.h TU/v/CmdPane.h TU/v/TU/v/CmdWindow.h \
	TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
FileSelection.o: TU/v/FileSelection.h TU/v/TU/v/ModalDialog.h \
	TU/v/TU/v/TU/v/Dialog.h TU/v/CmdPane.h TU/v/TU/v/CmdWindow.h \
	TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/Notify.h TU/v/Confirm.h
FrameCmd.o: FrameCmd_.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
Icon.o: TU/v/Icon.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h
LabelCmd.o: LabelCmd_.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
ListCmd.o: ListCmd_.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	vViewport_.h
MemoryDC.o: TU/v/MemoryDC.h TU/v/TU/v/XDC.h TU/v/TU/v/TU/v/DC.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h \
	TU/v/TU/v/TU/v/Colormap.h TU/v/CanvasPane.h TU/v/TU/v/TUv++.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
Menu.o: TU/v/TU/v/Menu.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/Bitmap.h
MenuButtonCmd.o: MenuButtonCmd_.h TU/v/TU/v/Menu.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
ModalDialog.o: TU/v/TU/v/ModalDialog.h TU/v/TU/v/TU/v/Dialog.h \
	TU/v/CmdPane.h TU/v/TU/v/CmdWindow.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
Notify.o: TU/v/Notify.h TU/v/TU/v/ModalDialog.h TU/v/TU/v/TU/v/Dialog.h \
	TU/v/CmdPane.h TU/v/TU/v/CmdWindow.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
Object.o: TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
Pane.o: TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	vGridbox_.h
RadioButtonCmd.o: TU/v/Bitmap.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	RadioButtonCmd_.h TU/v/TU/v/TUv++.h /home/ueshiba/include/TU/List++.h \
	TU/v/TU/v/TU/v/Widget-Xaw.h vGridbox_.h
ShmDC.o: TU/v/ShmDC.h TU/v/CanvasPaneDC.h TU/v/TU/v/XDC.h \
	TU/v/TU/v/TU/v/DC.h /home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h \
	TU/v/TU/v/TU/v/Colormap.h TU/v/CanvasPane.h TU/v/TU/v/TUv++.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/TU/v/Menu.h
SliderCmd.o: SliderCmd_.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	vSlider_.h vGridbox_.h
TUv++.inst.o: TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	/home/ueshiba/include/TU/List++.cc
TextInCmd.o: TextInCmd_.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	vTextField_.h
Timer.o: TU/v/Timer.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/App.h
ToggleButtonCmd.o: ToggleButtonCmd_.h TU/v/TU/v/TUv++.h \
	TU/v/TU/v/TU/v/Colormap.h /home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	TU/v/Bitmap.h
Widget-Xaw.o: TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h \
	vGridbox_.h vTextField_.h vViewport_.h
Window.o: TU/v/App.h TU/v/TU/v/TUv++.h TU/v/TU/v/TU/v/Colormap.h \
	/home/ueshiba/include/TU/Image++.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/List++.h TU/v/TU/v/TU/v/Widget-Xaw.h
XDC.o: TU/v/TU/v/XDC.h TU/v/TU/v/TU/v/DC.h \
	/home/ueshiba/include/TU/Geometry++.h \
	/home/ueshiba/include/TU/utility.h \
	/home/ueshiba/include/TU/Minimize++.h \
	/home/ueshiba/include/TU/Vector++.h \
	/home/ueshiba/include/TU/Array++.h /home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Image++.h /home/ueshiba/include/TU/Manip.h \
	TU/v/TU/v/TU/v/Colormap.h
vCanvas.o: vCanvasP_.h vCanvas_.h
vGridbox.o: vGridboxP_.h vGridbox_.h
vScrollbar.o: vScrollbarP_.h vScrollbar_.h
vSlider.o: vSliderP_.h vSlider_.h
vTextField.o: vTextFieldP_.h vTextField_.h
vViewport.o: vScrollbar_.h vViewportP_.h vViewport_.h
