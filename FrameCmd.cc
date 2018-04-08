/*
 *  $Id$  
 */
#include "FrameCmd_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class FrameCmd							*
************************************************************************/
FrameCmd::FrameCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvFrameCmd", cmd)
{
    if (cmd.prop != 0)
    {
	const CmdDef*	subcmds = (const CmdDef*)cmd.prop;
	for (int i = 0; subcmds[i].type != C_EndOfList; ++i)
	    addCmd(Cmd::newCmd(*this, subcmds[i]));
    }
}

FrameCmd::~FrameCmd()
{
    for (Cmd* vcmd; (vcmd = detachCmd()) != 0; )
	delete vcmd;
}

const Object::Widget&
FrameCmd::widget() const
{
    return _widget;
}

}
}
