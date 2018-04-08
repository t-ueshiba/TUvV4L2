/*
 *  $Id$  
 */
#include "TU/v/CanvasPaneDC.h"
#include "vCanvas_.h"
#include "vViewport_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{m_X0125, m_X025 = 32400, m_X05, m_X10, m_X15, m_X20, m_X40, m_X80};

static MenuDef zoomMenu[] =
{
    {"x0.125", m_X0125, false, noSub},
    {"x0.25",  m_X025,  false, noSub},
    {"x0.5",   m_X05,   false, noSub},
    {"x1",     m_X10,   false, noSub},
    {"x1.5",   m_X15,   false, noSub},
    {"x2",     m_X20,   false, noSub},
    {"x4",     m_X40,   false, noSub},
    {"x8",     m_X80,   false, noSub},
    EndOfMenu
};

/************************************************************************
*  Xt event handlers							*
************************************************************************/
void
EVcanvasPaneDC(::Widget widget, XtPointer client_data, XEvent* event,
	       Boolean* boolean)
{
    extern void		EVkeyPress(::Widget, XtPointer, XEvent*, Boolean*);
    CanvasPaneDC&	canvasPaneDC = *(CanvasPaneDC*)client_data;
    switch (event->type)
    {
      case ButtonPress:
	switch (event->xbutton.button)
	{
	  case Button1:
	    canvasPaneDC.callback(Id_MouseButton1Press,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	    break;
	  case Button2:
	    canvasPaneDC.callback(Id_MouseButton2Press,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	    break;
	  case Button3:
	    canvasPaneDC._u_last = canvasPaneDC.dev2logU(event->xbutton.x);
	    canvasPaneDC._v_last = canvasPaneDC.dev2logU(event->xbutton.y);
	    canvasPaneDC.callback(Id_MouseButton3Press,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	    break;
	}
	break;
	
      case ButtonRelease:
	switch (event->xbutton.button)
	{
	  case Button1:
	    canvasPaneDC.callback(Id_MouseButton1Release,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	    break;
	  case Button2:
	    canvasPaneDC.callback(Id_MouseButton2Release,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	    break;
	  case Button3:
	    canvasPaneDC.callback(Id_MouseButton3Release,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	    break;
	}
	break;
	
      case MotionNotify:
	if (event->xmotion.state & Button1Mask)
	    canvasPaneDC.callback(Id_MouseButton1Drag,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	else if (event->xmotion.state & Button2Mask)
	    canvasPaneDC.callback(Id_MouseButton2Drag,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	else if (event->xmotion.state & Button3Mask)
	    canvasPaneDC.callback(Id_MouseButton3Drag,
				  CmdVal(event->xbutton.x, event->xbutton.y));
	else
	    canvasPaneDC.callback(Id_MouseMove,
				  CmdVal(event->xbutton.x, event->xbutton.y));
        break;
      
      case EnterNotify:
	canvasPaneDC.callback(Id_MouseEnterFocus, 0);
	break;

      case LeaveNotify:
	canvasPaneDC.callback(Id_MouseLeaveFocus, 0);
	break;

      case Expose:
	while (XCheckWindowEvent(XtDisplay(widget), XtWindow(widget),
				 ExposureMask, event) == True)
	    ;		// Consume all the remained exposure events.
	canvasPaneDC.repaintAll();
	break;

      case KeyPress:
	EVkeyPress(widget, client_data, event, boolean);
	break;
    }
}

void
CBcanvasPaneDC(::Widget, XtPointer client_data, XtPointer)
{
    CanvasPaneDC&	canvasPaneDC = *(CanvasPaneDC*)client_data;
  // Invoke initialization routine specific to each DC.
    canvasPaneDC.initializeGraphics();
  // For some DC types, setting plane masks is only possible after realization.
    canvasPaneDC.setLayer(canvasPaneDC.getLayer());
  // Invoke initialization routine overwritten by the user.
    canvasPaneDC.canvasPane().initializeGraphics();
}

/************************************************************************
*  class CanvasPaneDC							*
************************************************************************/
/*
 *  Public member functions
 */
CanvasPaneDC::CanvasPaneDC(CanvasPane& parentCanvasPane,
			   size_t w, size_t h, float zoom)
    :Object(parentCanvasPane),
     XDC(w != 0 ? w : parentCanvasPane.widget().width(),
	 h != 0 ? h : parentCanvasPane.widget().height(),
	 zoom,
	 window().colormap(),
	 XtAllocateGC(window().widget(), 0, 0, 0, 0, 0)),
     _widget(XtVaCreateManagedWidget("TUvCanvasPaneDC",	   // widget name
				     vCanvasWidgetClass,   // widget class 
				     parent().widget(),    // parent widget
				     XtNwidth,		deviceWidth(),
				     XtNheight,		deviceHeight(),
				     XtNborderWidth,	0,
				   // Set window background.
				     XtNbackground,
				         colormap().getUnderlayPixel(u_char(0),
								     0, 0),
				     nullptr)),
     _popup(*this, zoomMenu),
     _u_last(0), _v_last(0)
{
    setDeviceSize();
    
    XtAddEventHandler(_widget,			// get mouse events
		      ButtonPressMask	|
		      ButtonReleaseMask	|
		      PointerMotionMask	|
		      EnterWindowMask	|
		      LeaveWindowMask	|
		      ExposureMask      |
		      KeyPressMask,
		      FALSE, EVcanvasPaneDC, (XtPointer)this);
    XtAddCallback(_widget, XtNginitCallback, CBcanvasPaneDC, (XtPointer)this);
}

CanvasPaneDC::~CanvasPaneDC()
{
}

const Object::Widget&
CanvasPaneDC::widget() const
{
    return _widget;
}

DC&
CanvasPaneDC::setSize(size_t width, size_t height, float zoom)
{
    XDC::setSize(width, height, zoom);
  // Viewport の中でこの widget を小さくするとき, 以前描画したものの残
  // 骸が余白に残るのは見苦しいので、widget 全体をクリアしておく。また、
  // 直接 graphic hardware にアクセスする API （XIL など）と実行順序が
  // 入れ替わることを防ぐため、XSync() を呼ぶ（XDC.cc 参照）。
    XClearWindow(colormap().display(), drawable());
    XSync(colormap().display(), False);
    setDeviceSize();
    return *this;
}

void
CanvasPaneDC::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      default:
	parent().callback(id, val);
	return;

      case m_X0125:
	*this << x0_125;
	break;
      case m_X025:
	*this << x0_25;
	break;
      case m_X05:
	*this << x0_5;
	break;
      case m_X10:
	*this << x1;
	break;
      case m_X15:
	*this << x1_5;
	break;
      case m_X20:
	*this << x2;
	break;
      case m_X40:
	*this << x4;
	break;
      case m_X80:
	*this << x8;
	break;
    }

    canvasPane().moveDC(log2devU(_u_last), log2devV(_v_last));
    *this << TU::v::repaintAll;
}

void
CanvasPaneDC::grabKeyboard() const
{
    XtGrabKeyboard(_widget, FALSE, GrabModeAsync, GrabModeAsync, CurrentTime);
}

/*
 *  Protected member functions
 */
Drawable
CanvasPaneDC::drawable() const
{
    return XtWindow(_widget);
}

void
CanvasPaneDC::initializeGraphics()
{
}

DC&
CanvasPaneDC::repaintUnderlay()
{
    canvasPane().repaintUnderlay();
    return *this;
}

DC&
CanvasPaneDC::repaintOverlay()
{
    canvasPane().repaintOverlay();
    return *this;
}

/*
 *  Private member functions
 */
void
CanvasPaneDC::setDeviceSize()
{
    XtVaSetValues(canvasPane().widget(),
		  XtNchildMinWidth,	deviceWidth(),
		  XtNchildMinHeight,	deviceHeight(),
		  nullptr);
}

size_t
CanvasPaneDC::realWidth() const		// widget width: for XDC::clear()
{
    return widget().width();
}

size_t
CanvasPaneDC::realHeight() const	// widget height: for XDC::clear()
{
    return widget().height();
}

}
}
