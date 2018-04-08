/*
 *  $Id$  
 */
#include "ChoiceFrameCmd_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ChoiceFrameCmd							*
************************************************************************/
ChoiceFrameCmd::ChoiceFrameCmd(Object& parentObject, const CmdDef& cmd)
    :FrameCmd(parentObject, cmd)
{
    setValue(cmd.val);
}

ChoiceFrameCmd::~ChoiceFrameCmd()
{
}

void
ChoiceFrameCmd::callback(CmdId nid, CmdVal nval)
{
    if (findChild(nid) != 0)
    {
	setValue(nid);
	parent().callback(id(), getValue());
    }
    else
	parent().callback(nid, nval);
}

CmdVal
ChoiceFrameCmd::getValue() const
{
    const Cmd*	vcmd = findChildWithNonZeroValue();
    return (vcmd != 0 ? vcmd->id() : M_None);
}

void
ChoiceFrameCmd::setValue(CmdVal val)
{
    Cmd*	cvcmd = findChild(getValue());	// current command chosen
    Cmd*	nvcmd = findChild(val);		// next command chosen
    if (nvcmd != cvcmd)
    {
	if (cvcmd != 0)
	    cvcmd->setValue(0);
	if (nvcmd != 0)
	    nvcmd->setValue(1);
    }
}

}
}
