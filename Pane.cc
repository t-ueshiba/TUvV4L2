/*
 *  $Id$  
 */
#include "TU/v/TUv++.h"
#include "vGridbox_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class Pane								*
************************************************************************/
Pane::Pane(Window& parentWin)
    :Object(parentWin.paned())
{
    window().addPane(*this);
}

Pane::~Pane()
{
    window().detachPane(*this);
}

void
Pane::place(size_t left, size_t top, size_t width, size_t height)
{
    XtVaSetValues(widget(), XtNgridx, left, XtNgridy, top,
		  XtNgridWidth, width, XtNgridHeight, height, nullptr);
}

}
}
