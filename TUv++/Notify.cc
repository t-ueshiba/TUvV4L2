/*
 *  $Id: Notify.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/Notify.h"

namespace TU
{
namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_Message, c_OK};

static CmdDef Cmds[] =
{
    {C_Label,  c_Message, 0, "",   noProp, CA_NoBorder,	     1, 0, 1, 1, 20},
    {C_Button, c_OK,	  0, "OK", noProp, CA_DefaultButton, 0, 0, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  class Notify								*
************************************************************************/
Notify::Notify(Window&	parentWindow)
    :ModalDialog(parentWindow, "Notify", Cmds)
{
}

Notify::~Notify()
{
}

void
Notify::show()
{
    pane().setString(c_Message, str().data());
    ModalDialog::show();
}

void
Notify::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case c_OK:
	hide();
	break;
    }
}

}
}
