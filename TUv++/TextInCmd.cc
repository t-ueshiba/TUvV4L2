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
 *  $Id: TextInCmd.cc,v 1.5 2008-09-10 05:12:18 ueshiba Exp $  
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
    XtVaSetValues(_widget, XtNstring, str, NULL);
}

const char*
TextInCmd::getString() const
{
    const char*	str;
    XtVaGetValues(_widget, XtNstring, &str, NULL);
    return str;
}

}
}
