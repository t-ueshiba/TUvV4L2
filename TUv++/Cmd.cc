/*
 *  $Id: Cmd.cc,v 1.2 2002-07-25 02:38:10 ueshiba Exp $
 */
#include "TU/v/TUv++.h"
#include "LabelCmd_.h"
#include "SliderCmd_.h"
#include "FrameCmd_.h"
#include "ButtonCmd_.h"
#include "ToggleButtonCmd_.h"
#include "MenuButtonCmd_.h"
#include "ChoiceMenuButtonCmd_.h"
#include "RadioButtonCmd_.h"
#include "ChoiceFrameCmd_.h"
#include "ListCmd_.h"
#include "TextInCmd_.h"
#include <stdexcept>

namespace TU
{
namespace v
{
/************************************************************************
*  Default callback for Cmd						*
************************************************************************/
static void
CBvCmd(Widget, XtPointer This, XtPointer)
{
    Cmd*	vcmd = (Cmd*)This;
    vcmd->callback(vcmd->id(), vcmd->getValue());
}

/************************************************************************
*  class Cmd								*
************************************************************************/
Cmd::Cmd(Object& parentObject, CmdId id)
    :Object(parentObject), _id(id)
{
}

Cmd::~Cmd()
{
}

Cmd*
Cmd::newCmd(Object& parentObject, const CmdDef& cmd)
{
    switch (cmd.type)
    {
      case C_Button:		// Button
	return new ButtonCmd(parentObject, cmd);

      case C_ToggleButton:	// Toggle button
	return new ToggleButtonCmd(parentObject, cmd);

      case C_RadioButton:	// Radio button
	return new RadioButtonCmd(parentObject, cmd);

      case C_Frame:		// General purpose frame
	return new FrameCmd(parentObject, cmd);

      case C_ChoiceFrame:	// Choice frame
	return new ChoiceFrameCmd(parentObject, cmd);

      case C_MenuButton:	// Menu button
	return new MenuButtonCmd(parentObject, cmd);

      case C_ChoiceMenuButton:	// Choie menu button
	return new ChoiceMenuButtonCmd(parentObject, cmd);

      case C_Icon:		// Display only icon
      case C_Label:		// Regular text label
	return new LabelCmd(parentObject, cmd);
	    
      case C_Slider:		// Slider
	return new SliderCmd(parentObject, cmd);

      case C_List:		// List
	return new ListCmd(parentObject, cmd);

      case C_TextIn:		// TextIn
	return new TextInCmd(parentObject, cmd);
    }

    throw std::domain_error("TU::v::Cmd::newCmd: Unknown command type!!");

    return 0;
}

CmdVal
Cmd::getValue() const
{
    return 0;
}

void
Cmd::setValue(CmdVal)
{
}

const char*
Cmd::getString() const
{
    const char*	str;
    XtVaGetValues(widget(), XtNlabel, &str, NULL);
    return str;
}

void
Cmd::setString(const char* str)
{
    XtVaSetValues(widget(), XtNlabel, str, NULL);
}

void
Cmd::setProp(void*)
{
}

void
Cmd::setDefaultCallback(const Widget& widget)
{
    XtAddCallback(widget, XtNcallback, CBvCmd, (XtPointer)this);
}

}
}
