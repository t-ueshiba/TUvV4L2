/*
 *  $Id: CanvasPane.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/CanvasPane.h"
#include "vViewport_.h"
#include "vGridbox_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPane							*
************************************************************************/
CanvasPane::CanvasPane(Window& parentWin, u_int devWidth, u_int devHeight)
    :Pane(parentWin),
     _widget(XtVaCreateManagedWidget("TUvCanvasPane",
				     vViewportWidgetClass,  // widget class 
				     parent().widget(),     // parent widget
				     XtNallowHoriz,	TRUE,
				     XtNallowVert,	TRUE,
				     XtNwidth,	(devWidth != 0 ? devWidth :
						 parent().widget().width()),
				     XtNheight,	(devHeight != 0 ? devHeight :
						 parent().widget().height()),
				     XtNuseBottom,	TRUE,
				     XtNuseRight,	TRUE,

				   // Expand/shrink according to the width/
				   // height of Paned.
				     XtNweightx,	1,
				     XtNweighty,	1,
				     NULL))
{
}

CanvasPane::~CanvasPane()
{
}

const Object::Widget&
CanvasPane::widget() const
{
    return _widget;
}

CanvasPane&
CanvasPane::canvasPane()
{
    return *this;
}

void
CanvasPane::initializeGraphics()
{
}

void
CanvasPane::repaintUnderlay(int /* u */, int /* v */,
			    int /* width */, int /* height */)
{
}

void
CanvasPane::repaintOverlay(int /* u */, int /* v */,
			   int /* width */, int /* height */)
{
}

void
CanvasPane::moveDC(int u, int v)
{
    vViewportSetCoordinates(_widget,
			    u - _widget.width()/2, v - _widget.height()/2);
}

}
}
