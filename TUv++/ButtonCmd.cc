/*
 *  $Id: ButtonCmd.cc,v 1.2 2002-07-25 02:38:09 ueshiba Exp $
 */
#include "ButtonCmd_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ButtonCmd							*
************************************************************************/
ButtonCmd::ButtonCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvButtonCmd", cmd),
     _bitmap(cmd.prop == noProp ? 0 :
	     new Bitmap(window().colormap(), (u_char*)cmd.prop))
{
    setDefaultCallback(_widget);
    if (_bitmap != 0)
	XtVaSetValues(_widget, XtNbitmap, _bitmap->xpixmap(), NULL);
}

ButtonCmd::~ButtonCmd()
{
    delete _bitmap;
}

const Object::Widget&
ButtonCmd::widget() const
{
    return _widget;
}

}
}
