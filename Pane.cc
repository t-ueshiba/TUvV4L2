/*
 *  $Id: Pane.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
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
Pane::place(u_int left, u_int top, u_int width, u_int height)
{
    XtVaSetValues(widget(), XtNgridx, left, XtNgridy, top,
		  XtNgridWidth, width, XtNgridHeight, height, NULL);
}

}
}
