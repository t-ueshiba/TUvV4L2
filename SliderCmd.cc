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
 *  $Id: SliderCmd.cc,v 1.7 2009-01-03 08:51:41 ueshiba Exp $  
 */
#include "SliderCmd_.h"
#include "vSlider_.h"
#include "vGridbox_.h"
#include <X11/Xaw3d/Label.h>
#include <sstream>
#include <iomanip>

namespace TU
{
namespace v
{
/************************************************************************
*  callbacks for class SliderCmd					*
************************************************************************/
static void
CBsliderCmdJumpProc(Widget, XtPointer This, XtPointer pc_ptr)
{
    SliderCmd*	vSliderCmd = (SliderCmd*)This;
    float	percent = *(float*)pc_ptr;	// get the percent back
    vSliderCmd->setPercent(percent);
    vSliderCmd->callback(vSliderCmd->id(), vSliderCmd->getValue());
}

/************************************************************************
*  class SliderCmd							*
************************************************************************/
SliderCmd::SliderCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvSliderCmd", cmd),
     _title(XtVaCreateManagedWidget("TUvSliderCmd-title",
				    labelWidgetClass,
				    _widget,
				    XtNlabel,		cmd.title,
				    XtNborderWidth,	0,
				    XtNfill,		"none",
				    XtNgravity,		WestGravity,
				    XtNgridx,		0,
				    NULL)),
     _slider(XtVaCreateManagedWidget("TUvSliderCmd-slider",
				     sliderWidgetClass,
				     _widget,
				     XtNminimumThumb,	10,
				     XtNthickness,	20,	// how wide
				     XtNorientation,
				       (cmd.attrs & CA_Vertical ?
					XtorientVertical : XtorientHorizontal),
				     XtNlength,
				       (cmd.size != 0 ? cmd.size : 100),
				     XtNtopOfThumb,	0.0,
				     XtNshown,		0.0,
				     XtNfill,		"none",
				     XtNgravity,	WestGravity,
				     XtNgridx,		1,
				     NULL)),
     _text(XtVaCreateManagedWidget("TUvSliderCmd-text",
				   labelWidgetClass,
				   _widget,
				   XtNborderWidth,	1,
				   XtNfill,		"none",
				   XtNgravity,		WestGravity,
				   XtNgridx,		2,
				   NULL)),
     _min  (cmd.prop != 0 ? ((int*)cmd.prop)[0] :   0),
     _range(cmd.prop != 0 ? ((int*)cmd.prop)[1] : 100),
     _div  (cmd.prop != 0 ? ((int*)cmd.prop)[2] : 100),
     _val(cmd.val)
{
    XtVaSetValues(_slider, XtNgridx, 1, NULL);
    XtAddCallback(_slider, XtNjumpProc, CBsliderCmdJumpProc, this);
    
    setValue(CmdVal(_val, _div));
}

SliderCmd::~SliderCmd()
{
}

const Object::Widget&
SliderCmd::widget() const
{
    return _widget;
}

CmdVal
SliderCmd::getValue() const
{
    return CmdVal(_val, _div);
}

void
SliderCmd::setValue(CmdVal val)
{
    setValueInternal(val);
    float	percent = (_val - _min) / (float)_range;
    XawSliderSetThumb(_slider, percent, 0.0);
}

void
SliderCmd::setProp(void* prop)
{
    if (prop != 0)
    {
	int*	ip = (int*)prop;
	_min   = ip[0];
	_range = ip[1];
	_div   = ip[2];
    }
    else
    {
	_min   = 0;
	_range = 100;
	_div   = 1;
    }
}

void
SliderCmd::setPercent(float percent)
{
    setValueInternal(CmdVal(_min + int(percent * _range), _div));
}

void
SliderCmd::setValueInternal(CmdVal val)
{
    _val = int(val.f() * _div);
    std::ostringstream	s;
    if (_div == 1)
	s << std::setw(4) << _val;
    else
	s << std::setw(4) << (float)_val / (float)_div;
    XtVaSetValues(_text, XtNlabel, s.str().c_str(), NULL);
}

}
}
