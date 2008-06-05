/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: Window.cc,v 1.6 2008-06-05 02:26:24 ueshiba Exp $
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
    XtVaGetValues(widget(), XtNoverrideRedirect, &overrideRedirect, NULL);
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

