/*
 *  $Id: ChoiceMenuButtonCmd.cc,v 1.2 2002-07-25 02:38:10 ueshiba Exp $
 */
#include "ChoiceMenuButtonCmd_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ChoiceMenuButtonCmd						*
************************************************************************/
ChoiceMenuButtonCmd::ChoiceMenuButtonCmd(Object& parentObject,
					 const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvChoiceMenuButtonCmd", cmd),
     _menu(*this, (MenuDef*)cmd.prop)
{
    addCmd(&_menu);
    setValue(getValue());
}

ChoiceMenuButtonCmd::~ChoiceMenuButtonCmd()
{
    detachCmd();
}

const Object::Widget&
ChoiceMenuButtonCmd::widget() const
{
    return _widget;
}

void
ChoiceMenuButtonCmd::callback(CmdId nid, CmdVal nval)
{
    if (nid == _menu.id())
    {
	setValue(nval);
	parent().callback(id(), getValue());
    }
    else
	parent().callback(nid, nval);
}

CmdVal
ChoiceMenuButtonCmd::getValue() const
{
    return _menu.getValue();
}

void
ChoiceMenuButtonCmd::setValue(CmdVal val)
{
    _menu.setValue(val);
}

}
}
