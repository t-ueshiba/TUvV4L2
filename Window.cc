/*
 *  $Id$  
 */
#include "TU/v/App.h"
#include <X11/Shell.h>
#include <stdexcept>

namespace TU
{
namespace v
{
/************************************************************************
*  class Window							*
************************************************************************/
Window::Window(Window& parentWindow)
    :Object(parentWindow), _windowList(), _paneList()
{
    if (&parent() != this)
	parent().window().addWindow(*this);
}

Window::~Window()
{
    if (&parent() != this)
	parent().window().detachWindow(*this);
}

Colormap&
Window::colormap()
{
    return app().colormap();
}

void
Window::show()
{
    XtPopup(widget(), XtGrabNone);
}

void
Window::hide()
{
    XtPopdown(widget());
}

bool
Window::isFullScreen() const
{
    Boolean	overrideRedirect;
    XtVaGetValues(widget(), XtNoverrideRedirect, &overrideRedirect, nullptr);
    return (overrideRedirect == TRUE);
}

/*
 *  protected member funtions
 */
Window&
Window::window()
{
    return *this;
}

CanvasPane&
Window::canvasPane()
{
    throw std::domain_error("TU::v::Window::canvasPane(): No canvasPane found!!");
    return canvasPane();
}

Object&
Window::paned()
{
    return *this;
}

}
}

