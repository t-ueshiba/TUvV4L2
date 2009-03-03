/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id: Cmd.cc,v 1.6 2009-03-03 00:59:47 ueshiba Exp $  
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
    XtVaGetValues(widget(), XtNlabel, &str, Null);
    return str;
}

void
Cmd::setString(const char* str)
{
    XtVaSetValues(widget(), XtNlabel, str, Null);
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
