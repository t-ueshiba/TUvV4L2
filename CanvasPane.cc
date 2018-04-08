/*
 *  $Id$  
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
CanvasPane::CanvasPane(Window& parentWin, size_t devWidth, size_t devHeight)
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
				     nullptr))
{
    if (window().isFullScreen())
	XtVaSetValues(_widget, XtNborderWidth, 0, nullptr);
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
CanvasPane::repaintUnderlay()
{
}

void
CanvasPane::repaintOverlay()
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
