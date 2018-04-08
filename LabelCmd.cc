/*
 *  $Id$  
 */
#include "LabelCmd_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class LabelCmd							*
************************************************************************/
LabelCmd::LabelCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvLabelCmd", cmd)
{
}

LabelCmd::~LabelCmd()
{
}

const Object::Widget&
LabelCmd::widget() const
{
    return _widget;
}

}
}
