/*
 *  $Id$  
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
	XtVaSetValues(_widget, XtNbitmap, _bitmap->xpixmap(), nullptr);
}

ButtonCmd::~ButtonCmd()
{
    removeDefaultCallback(_widget);
    delete _bitmap;
}

const Object::Widget&
ButtonCmd::widget() const
{
    return _widget;
}

}
}
