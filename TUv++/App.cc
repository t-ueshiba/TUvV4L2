/*
 *  $Id: App.cc,v 1.3 2004-03-08 02:12:26 ueshiba Exp $
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
    XtVaGetValues(widget, XtNvisual, &visual, NULL);
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
    XtVaGetValues(menuItem, XtNwidth, &width, NULL);

    if (event->xmotion.x < width - (width / 3) || event->xmotion.x >= width)
	return;

  /* Get submenu widget and pop it up. */
    char*	name;
    XtVaGetValues(menuItem, XtNlabel, &name, NULL);
    Widget	submenu = XtNameToWidget(menu, name);
    if (submenu != 0)
    {
	Position	x, y;
	XtTranslateCoords(menuItem, event->xmotion.x - 10, 0, &x, &y);
	XtVaSetValues(submenu, XtNx, x, XtNy, y, NULL);
	XtPopup(submenu, XtGrabExclusive);
    }
}

static int
xerrorHandler(Display* display, XErrorEvent* error)
{
    char	buf[129];
    XGetErrorText(display, error->error_code, buf, sizeof(buf)-1);
    throw std::runtime_error(buf);

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
    "*Slider.translations:\
	<BtnDown>:	StartScroll(Continuous) MoveThumb() NotifyThumb()\\n\
	<BtnMotion>:	MoveThumb() NotifyThumb()			\\n\
	<BtnUp>:	NotifyScroll(Proportional) EndScroll()",
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
			       NULL)),
     _colormap(XtDisplay(_widget), vinfoOfWidget(_widget)),
     _active(0)
{
    XtVaSetValues(_widget, XtNcolormap, (::Colormap)_colormap, NULL);
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
    XtDestroyWidget(_widget);
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
App::run()
{
    _active = true;
    while (_active)
    {
	XEvent	event;
	
	XtAppNextEvent(_appContext, &event);
	XtDispatchEvent(&event);
    }
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
