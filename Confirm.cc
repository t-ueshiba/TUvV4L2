/*
 *  $Id$  
 */
#include "TU/v/Confirm.h"

namespace TU
{
namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_Message, c_Cancel, c_OK};

static CmdDef Cmds[] =
{
    {C_Label,  c_Message, 0, "",       noProp, CA_NoBorder,	 1,0,2,1, 20},
    {C_Button, c_Cancel,  0, "Cancel", noProp, CA_DefaultButton, 0,0,1,1, 0},
    {C_Button, c_OK,	  0, "OK",     noProp, CA_None,		 0,1,1,1, 0},
    EndOfCmds
};

/************************************************************************
*  class Confirm							*
************************************************************************/
Confirm::Confirm(Window& parentWindow)
    :ModalDialog(parentWindow, "Confirm", Cmds), _ok(false)
{
}

Confirm::~Confirm()
{
}

bool
Confirm::ok()
{
    pane().setString(c_Message, str().c_str());
    show();
    return _ok;
}

void
Confirm::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case c_Cancel:
	_ok = false;
	hide();
	break;
      case c_OK:
	_ok = true;
	hide();
	break;
    }
}

}
}
