/*
 *  $Id: MenuButtonCmd.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "MenuButtonCmd_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MenuButtonCmd							*
************************************************************************/
MenuButtonCmd::MenuButtonCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvMenuButtonCmd", cmd),
     _menu(*this, (MenuDef*)cmd.prop)
{
    addCmd(&_menu);
}

MenuButtonCmd::~MenuButtonCmd()
{
    detachCmd();
}

const Object::Widget&
MenuButtonCmd::widget() const
{
    return _widget;
}

}
}
