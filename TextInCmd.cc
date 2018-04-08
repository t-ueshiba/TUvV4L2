/*
 *  $Id$  
 */
#include "TextInCmd_.h"
#include "vTextField_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  Default callback for Cmd						*
************************************************************************/
static void
CBtextInCmd(Widget, XtPointer This, XtPointer)
{
    TextInCmd*	vTextInCmd = (TextInCmd*)This;
    vTextInCmd->callback(vTextInCmd->id(), vTextInCmd->getValue());
}

/************************************************************************
*  class TextInCmd							*
************************************************************************/
TextInCmd::TextInCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvTextInCmd", cmd)
{
    XtAddCallback(_widget, XtNactivateCallback, CBtextInCmd, this);
}

TextInCmd::~TextInCmd()
{
}

const Object::Widget&
TextInCmd::widget() const
{
    return _widget;
}

void
TextInCmd::setString(const char* str)
{
    XtVaSetValues(_widget, XtNstring, str, nullptr);
}

const char*
TextInCmd::getString() const
{
    const char*	str;
    XtVaGetValues(_widget, XtNstring, &str, nullptr);
    return str;
}

}
}
