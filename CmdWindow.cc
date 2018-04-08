/*
 *  $Id$  
 */
#include "TU/v/CmdWindow.h"
#include "TU/v/App.h"
#include "vGridbox_.h"
#include <X11/Shell.h>
#include <X11/keysym.h>

namespace TU
{
namespace v
{
/************************************************************************
*  static functions							*
************************************************************************/
static XVisualInfo
selectVisual(Display* display, int screen,
	     Colormap::Mode mode, size_t overlayDepth)
{
    XVisualInfo		vinfo_template[5];
    long		vinfo_mask[5];
    int			n = 0;

    if (mode == Colormap::IndexedColor)
    {
	vinfo_template[n].c_class = DirectColor;
	vinfo_template[n].depth	  = 24;
	vinfo_template[n].screen  = screen;
	vinfo_mask[n++] = VisualClassMask | VisualDepthMask | VisualScreenMask;

	vinfo_template[n].c_class = PseudoColor;
	vinfo_template[n].screen  = screen;
	vinfo_mask[n++] = VisualClassMask | VisualScreenMask;

	vinfo_template[n].c_class = GrayScale;
	vinfo_template[n].screen  = screen;
	vinfo_mask[n++] = VisualClassMask | VisualScreenMask;
    }
    else					// RGBColor mode.
    {
	if (overlayDepth != 0)
	{
	    vinfo_template[n].c_class = DirectColor;
	    vinfo_template[n].depth   = 24;
	    vinfo_template[n].screen  = screen;
	    vinfo_mask[n++] = VisualClassMask | VisualDepthMask |
			      VisualScreenMask;
	}
	else					// No overlay planes.
	{
	    vinfo_template[n].c_class = TrueColor;
	    vinfo_template[n].depth   = 24;
	    vinfo_template[n].screen  = screen;
	    vinfo_mask[n++] = VisualClassMask | VisualDepthMask |
			      VisualScreenMask;

	    vinfo_template[n].c_class = TrueColor;
	    vinfo_template[n].depth   = 16;
	    vinfo_template[n].screen  = screen;
	    vinfo_mask[n++] = VisualClassMask | VisualDepthMask |
			      VisualScreenMask;

	    vinfo_template[n].c_class = PseudoColor;
	    vinfo_template[n].screen  = screen;
	    vinfo_mask[n++] = VisualClassMask | VisualScreenMask;

	    vinfo_template[n].c_class = TrueColor;
	    vinfo_template[n].screen  = screen;
	    vinfo_mask[n++] = VisualClassMask | VisualScreenMask;

	    vinfo_template[n].c_class = StaticGray;
	    vinfo_template[n].screen  = screen;
	    vinfo_mask[n++] = VisualClassMask | VisualScreenMask;
	}
    }
    
    XVisualInfo	vinfo_return;
    for (int i = 0; i < n; ++i)
    {
	int		nvinfo;
	XVisualInfo*	vinfo = XGetVisualInfo(display, vinfo_mask[i],
					       &vinfo_template[i], &nvinfo);
	if (nvinfo != 0)
	{
	    vinfo_return = vinfo[0];
	    for (int j = 1; j < nvinfo; ++j)
		if (vinfo[j].colormap_size > vinfo_return.colormap_size)
		    vinfo_return = vinfo[j];
	    XFree((char*)vinfo);

	    return vinfo_return;
	}
    }

    throw std::runtime_error("No appropriate visual!!");
    return vinfo_return;
}

/************************************************************************
*  Xt callbacks written in C						*
************************************************************************/
void
EVcmdWindow(::Widget widget, XtPointer cmdWindowPtr, XEvent* event, Boolean*)
{
    static int	inUse = 0;
    CmdWindow*	cmdWindow = (CmdWindow*)cmdWindowPtr;
    Atom	wm_protocols = XInternAtom(XtDisplay(widget),
					   "WM_PROTOCOLS", False);    
    if (inUse || event->type != ClientMessage ||
	event->xclient.message_type != wm_protocols)
	return;
    inUse = 1;
    cmdWindow->app().exit();
    inUse = 0;
}

/************************************************************************
*  class CmdWindow							*
************************************************************************/
CmdWindow::CmdWindow(Window&		parentWindow,
		     const char*	myName,
		     Colormap::Mode	mode,
		     size_t		resolution,
		     size_t		underlayCmapDim,
		     size_t		overlayDepth,
		     int		screen,
		     bool		fullScreen)
    :Window(parentWindow),
     _colormap(parentWindow.colormap().display(),
	       selectVisual(parentWindow.colormap().display(),
			    (screen < 0 ?
			     parentWindow.colormap().vinfo().screen : screen),
			    mode, overlayDepth),
	       mode, resolution, underlayCmapDim, overlayDepth, 6, 9, 4),
     _widget(XtVaCreatePopupShell("TUvCmdWindow",
				  topLevelShellWidgetClass,
				  parent().widget(),
				  XtNtitle,		myName,
				  XtNallowShellResize,	TRUE,
				  XtNscreen,	ScreenOfDisplay(
						    _colormap.display(),
						    _colormap.vinfo().screen),
				  XtNvisual,	_colormap.vinfo().visual,
				  XtNdepth,	_colormap.vinfo().depth,
				  XtNcolormap,	(::Colormap)_colormap,
				  nullptr)),
     _paned(*this)
{
    if (fullScreen)
	XtVaSetValues(_widget,
		      XtNallowShellResize,  FALSE,
		      XtNborderWidth,	    0,	
		      XtNx,		    0,
		      XtNy,		    0,
		      XtNwidth,		    WidthOfScreen(XtScreen(_widget)),
		      XtNheight,	    HeightOfScreen(XtScreen(_widget)),
		      XtNoverrideRedirect,  TRUE,
		      nullptr);
    XtAddEventHandler(_widget, 0L, TRUE, EVcmdWindow, this);
}

CmdWindow::CmdWindow(Window&		parentWindow,
		     const char*	myName,
		     const XVisualInfo*	xvinfo,
		     Colormap::Mode	mode,
		     size_t		resolution,
		     size_t		underlayCmapDim,
		     size_t		overlayDepth,
		     bool		fullScreen)
    :Window(parentWindow),
     _colormap(parentWindow.colormap().display(), 
	       (xvinfo != 0 ? *xvinfo : parentWindow.colormap().vinfo()),
	       mode, resolution, underlayCmapDim, overlayDepth, 6, 9, 4),
     _widget(XtVaCreatePopupShell("TUvCmdWindow",
				  topLevelShellWidgetClass,
				  parent().widget(),
				  XtNtitle,		myName,
				  XtNallowShellResize,	TRUE,
				  XtNscreen,	ScreenOfDisplay(
						    _colormap.display(),
						    _colormap.vinfo().screen),
				  XtNvisual,	_colormap.vinfo().visual,
				  XtNdepth,	_colormap.vinfo().depth,
				  XtNcolormap,	(::Colormap)_colormap,
				  nullptr)),
     _paned(*this)
{
    if (fullScreen)
	XtVaSetValues(_widget,
		      XtNallowShellResize,  FALSE,
		      XtNborderWidth,	    0,	
		      XtNx,		    0,
		      XtNy,		    0,
		      XtNwidth,		    WidthOfScreen(XtScreen(_widget)),
		      XtNheight,	    HeightOfScreen(XtScreen(_widget)),
		      XtNoverrideRedirect,  TRUE,
		      nullptr);
    XtAddEventHandler(_widget, 0L, TRUE, EVcmdWindow, this);
}

CmdWindow::~CmdWindow()
{
}

const Object::Widget&
CmdWindow::widget() const
{
    return _widget;
}

Colormap&
CmdWindow::colormap()
{
    return _colormap;
}

void
CmdWindow::show()
{
    Window::show();
    app().addColormapWindow(*this);

  // Add the delete protocol.
    Atom	delete_window = XInternAtom(XtDisplay(_widget),
					    "WM_DELETE_WINDOW", False);
    XSetWMProtocols(XtDisplay(_widget), XtWindow(_widget),
		    &delete_window, 1);

    XRaiseWindow(XtDisplay(_widget), XtWindow(_widget));
}

/*
 *  protected functions
 */
Object&
CmdWindow::paned()
{
    return _paned;
}

/************************************************************************
*  Xt callbacks written in C						*
************************************************************************/
void
EVkeyPress(::Widget widget, XtPointer client_data, XEvent* event, Boolean*)
{
    KeySym	keysym;
    Modifiers	modifiers;
    XtTranslateKeycode(XtDisplay(widget), event->xkey.keycode,
		       event->xkey.state, &modifiers, &keysym);
    int	id = (keysym == XK_BackSpace ? '\b' :
	      keysym == XK_Tab	     ? '\t' :
	      keysym == XK_Linefeed  ? '\n' :
	      keysym == XK_Return    ? '\r' :
	      XK_space <= keysym && keysym <= XK_asciitilde ?
	      keysym - XK_space +  ' ' : 0);
    if (id != 0)
    {
	if (event->xkey.state & ControlMask)
	    id |= VKM_Ctrl;
	if (event->xkey.state & Mod1Mask)
	    id |= VKM_Alt;
	Object& object = *(Object*)client_data;
	object.callback(Id_KeyPress, id);
    }
}

/************************************************************************
*  class CmdWindow::Paned						*
************************************************************************/
CmdWindow::Paned::Paned(CmdWindow& parentCmdWindow)
    :Object(parentCmdWindow),
     _widget(XtVaCreateManagedWidget("TUvCmdWindow-Paned",
				     gridboxWidgetClass,
				     parent().widget(),
				     XtNdefaultDistance,	0,
				     nullptr))
{
    XtAddEventHandler(_widget, KeyPressMask, FALSE, EVkeyPress,
		      (XtPointer)this);
}

CmdWindow::Paned::~Paned()
{
}

const Object::Widget&
CmdWindow::Paned::widget() const
{
    return _widget;
}

}
}
