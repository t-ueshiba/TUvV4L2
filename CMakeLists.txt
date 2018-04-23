#
#
#
cmake_minimum_required(VERSION 2.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(X11 REQUIRED)
#find_package(X11_Xaw3d REQUIRED)

include_directories(
    .
    ../TUTools++
    ${X11_INCLUDE_DIRS}
    ${Xaw3d_INCLUDE_DIRS})

set(sources
    App.cc
    Bitmap.cc
    ButtonCmd.cc
    CanvasPane.cc
    CanvasPaneDC.cc
    CanvasPaneDC3.cc
    ChoiceFrameCmd.cc
    ChoiceMenuButtonCmd.cc
    Cmd.cc
    CmdPane.cc
    CmdParent.cc
    CmdWindow.cc
    Colormap.cc
    Confirm.cc
    DC.cc
    DC3.cc
    Dialog.cc
    FileSelection.cc
    FrameCmd.cc
    Icon.cc
    LabelCmd.cc
    ListCmd.cc
    MemoryDC.cc
    Menu.cc
    MenuButtonCmd.cc
    ModalDialog.cc
    Notify.cc
    Object.cc
    Pane.cc
    RadioButtonCmd.cc
    ShmDC.cc
    SliderCmd.cc
    TUv++.inst.cc
    TextInCmd.cc
    Timer.cc
    ToggleButtonCmd.cc
    Widget-Xaw.cc
    Window.cc
    XDC.cc
    vCanvas.c
    vGridbox.c
    vSlider.c
    vTextField.c
    vViewport.c)

set(headers
    ButtonCmd_.h
    ChoiceFrameCmd_.h
    ChoiceMenuButtonCmd_.h
    FrameCmd_.h
    LabelCmd_.h
    ListCmd_.h
    MenuButtonCmd_.h
    RadioButtonCmd_.h
    SliderCmd_.h
    TextInCmd_.h
    ToggleButtonCmd_.h
    vCanvasP_.h
    vCanvas_.h
    vGridboxP_.h
    vGridbox_.h
    vSliderP_.h
    vSlider_.h
    vTextFieldP_.h
    vTextField_.h
    vViewportP_.h
    vViewport_.h)

add_library(TUv++ SHARED ${sources})

target_link_libraries(
    TUv++
    TUTools++
    ${X11_Xaw3d_LIB}
    ${X11_Xt_LIB}
    ${X11_Xmu_LIB}
    ${X11_LIBRARIES})

install(TARGETS TUv++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
