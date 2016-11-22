/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id$  
 */
#include "TU/v/App.h"
#include <stdexcept>
#include <X11/Shell.h>
#include <X11/Xaw3d/SimpleMenu.h>

namespace TU
{
namespace v
{
/************************************************************************
*  static functions							*
************************************************************************/
static XVisualInfo
vinfoOfWidget(Widget widget)
{
    Visual*		visual;
    XtVaGetValues(widget, XtNvisual, &visual, nullptr);
    XVisualInfo		vinfo_template;
    vinfo_template.visualid
	= XVisualIDFromVisual(visual != 0 ? visual :
			      DefaultVisualOfScreen(XtScreen(widget)));

    int			nvinfo;
    XVisualInfo*	vinfo = XGetVisualInfo(XtDisplay(widget),
					       VisualIDMask,
					       &vinfo_template, &nvinfo);
    if (nvinfo == 0)
    {
	throw std::runtime_error("vinfoOfWidget: failed to get XVisualInfo!!");
    }

    XVisualInfo		vinfo_return = vinfo[0];
    XFree((char*)vinfo);
    
    return vinfo_return;
}

static void
submenuPopup(Widget menu, XEvent* event, String*, Cardinal*)
{
  /* Get indivisual menu item currently active. */
    Widget	menuItem = XawSimpleMenuGetActiveEntry(menu);

    if (menuItem == 0 || event->type != MotionNotify)
	return;

  /* Check current mouse position relative to the above item. */
    Dimension	width;
    XtVaGetValues(menuItem, XtNwidth, &width, nullptr);

    if (event->xmotion.x < width - (width / 3) || event->xmotion.x >= width)
	return;

  /* Get submenu widget and pop it up. */
    char*	name;
    XtVaGetValues(menuItem, XtNlabel, &name, nullptr);
    Widget	submenu = XtNameToWidget(menu, name);
    if (submenu != 0)
    {
	Position	x, y;
	XtTranslateCoords(menuItem, event->xmotion.x - 10, 0, &x, &y);
	XtVaSetValues(submenu, XtNx, x, XtNy, y, nullptr);
	XtPopup(submenu, XtGrabExclusive);
    }
}

static int
xerrorHandler(Display* display, XErrorEvent* error)
{
    using namespace	std;
    
    char	buf[129];
    XGetErrorText(display, error->error_code, buf, sizeof(buf)-1);
  //throw runtime_error(buf);
    cerr << buf << endl;

    return 0;
}

/************************************************************************
*  class App								*
************************************************************************/
static String	fallback[] =
{
  // Important to have topShadow and bottomShadow be drawn correctly.
    "*beNiceToColormap:			false",

  // Set default font.
    "*font:				7x13bold",

  // Followings are same as vLightControlShadow and vDarkControlShadow.
    "*topShadowPixel:			#DEDEDEDEDEDE",
    "*bottomShadowPixel:		#7F7F7F7F7F7F",
    "*borderColor:			white",
    "*shadowWidth:			2",
    
  // Following corresponds to both vControlBG and vControlFace.
    "*background:			#B3B3B3B3B3B3",

    "*TUvCanvasPaneDC.translations:\
	<Btn3Down>:	XawPositionSimpleMenu(TUvMenu) MenuPopup(TUvMenu)",
    "*TUvMenu.translations: #override\
	<BtnMotion>:	highlight() SubmenuPopup()",
    "*SimpleMenu.translations:\
	<LeaveWindow>:	unhighlight() MenuPopdown()		\\n\
	<BtnMotion>:	highlight() SubmenuPopup()		\\n\
	<BtnUp>:	MenuPopdown() notify() unhighlight()",
    NULL
};
static XrmOptionDescRec	desc[] =
{
    {"-visual", "*visual", XrmoptionSepArg,    NULL},
    {"-depth",  "*depth",  XrmoptionSepArg,    NULL},
    {"-iconic", ".iconic", XrmoptionSkipNArgs, 0},
    {"-name",	".name",   XrmoptionSkipArg,   NULL},
    {"-title",	".title",  XrmoptionSkipArg,   NULL}
};

	
App::App(int& argc, char* argv[])
    :Window(*this),
     _widget(XtVaAppInitialize(&_appContext,
			       "TUvApp",	// application class
			       desc, XtNumber(desc),// user specified resources
			       &argc, argv,
			       fallback,
			       XtNmappedWhenManaged,	FALSE,	// invisible
			       XtNwidth,		1,
			       XtNheight,		1,	// very small
			       nullptr)),
     _colormap(XtDisplay(_widget), vinfoOfWidget(_widget)),
     _active(0)
{
    XtVaSetValues(_widget, XtNcolormap, (::Colormap)_colormap, nullptr);
    XtRealizeWidget(_widget);
    
  // Add the delete protocol.
    Atom	delete_window = XInternAtom(XtDisplay(_widget),
					    "WM_DELETE_WINDOW", False);
    XSetWMProtocols(XtDisplay(_widget), XtWindow(_widget), &delete_window, 1);

  // Add actions.
    XtActionsRec	actions[] =
			{
			    {"SubmenuPopup", (XtActionProc)submenuPopup}
			};
    XtAppAddActions(_appContext, actions, XtNumber(actions));
    
  // Add error handlers.
    XSetErrorHandler((XErrorHandler)xerrorHandler);
}

App::~App()
{
#ifndef DESTROY_WIDGET
  //XtDestroyWidget(_widget);
#endif
}

const Object::Widget&
App::widget() const
{
    return _widget;
}

Colormap&
App::colormap()
{
    return _colormap;
}

void
App::callback(CmdId, CmdVal)
{
}

void
App::step()
{
    XEvent	event;
	
    XtAppNextEvent(_appContext, &event);
    XtDispatchEvent(&event);
}

void
App::run()
{
    _active = true;
    while (_active)
	step();
}

void
App::exit()
{
    _active = false;
}

void
App::addColormapWindow(const Window& vWindow) const
{
    ::Window*	windows = 0;
    int		nwindows = 0;
    XGetWMColormapWindows(XtDisplay(_widget), XtWindow(_widget),
			  &windows, &nwindows);

    ::Window*	windows_new = new ::Window[nwindows+1];
    for (int i = 0; i < nwindows; ++i)
	windows_new[i] = windows[i];
    windows_new[nwindows] = XtWindow(vWindow.widget());
    XSetWMColormapWindows(XtDisplay(_widget), XtWindow(_widget),
			  windows_new, nwindows+1);
    delete [] windows_new;
    if (nwindows != 0)
	XFree(windows);
}

App&
App::app()
{
    return *this;
}

}
}
