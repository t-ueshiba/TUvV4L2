/*
 *  $Id: CmdPane.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/CmdPane.h"
#include "vGridbox_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CmdPane							*
************************************************************************/
CmdPane::CmdPane(Window& parentWindow, const CmdDef cmd[])
    :Pane(parentWindow),
     _widget(XtVaCreateManagedWidget("TUvCmdPane",
				     gridboxWidgetClass,	// widget class
				     parent().widget(),
				   // Expand/shrink according to the width
				   // of Paned.
				     XtNweightx,		1,
				     NULL))
{
    for (int i = 0; cmd[i].type != C_EndOfList; ++i)
	addCmd(Cmd::newCmd(*this, cmd[i]));
}

CmdPane::~CmdPane()
{
    for (Cmd* vcmd; (vcmd = detachCmd()) != 0; )
	delete vcmd;
}

const Object::Widget&
CmdPane::widget() const
{
    return _widget;
}

}
}
