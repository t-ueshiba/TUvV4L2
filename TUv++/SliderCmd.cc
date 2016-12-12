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
 *  $Id$  
 */
#include "SliderCmd_.h"
#include "vSlider_.h"
#include "vGridbox_.h"
#include <X11/Xaw3d/Label.h>
#include <cmath>
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
				    XtNgravity,		EastGravity,
				    XtNgridx,		0,
				    XtNweightx,		1,
				    nullptr)),
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
				     XtNgravity,	EastGravity,
				     XtNgridx,		1,
				     XtNweightx,	0,
				     nullptr)),
     _text(XtVaCreateManagedWidget("TUvSliderCmd-text",
				   labelWidgetClass,
				   _widget,
				   XtNborderWidth,	1,
				   XtNfill,		"none",
				   XtNgravity,		EastGravity,
				   XtNgridx,		2,
				   XtNweightx,		0,
				   nullptr)),
     _min (cmd.prop ? static_cast<const float*>(cmd.prop)[0] : 0),
     _max (cmd.prop ? static_cast<const float*>(cmd.prop)[1] : 1),
     _step(cmd.prop ? static_cast<const float*>(cmd.prop)[2] : 0),
     _val (cmd.val)
{
    XtVaSetValues(_slider, XtNgridx, 1, nullptr);
    XtAddCallback(_slider, XtNjumpProc, CBsliderCmdJumpProc, this);
    
    setValue(_val);
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
    return _val;
}

void
SliderCmd::setValue(CmdVal val)
{
    setValueInternal(val.f());
    float	percent = (_val.f() - _min) / (_max - _min);
    vSliderSetThumb(_slider, percent, 0.0);
}

void
SliderCmd::setProp(const void* prop)
{
    if (prop)
    {
	const auto	fp = static_cast<const float*>(prop);
	_min  = fp[0];
	_max  = fp[1];
	_step = fp[2];
    }
    else
    {
	_min  = 0;
	_max  = 100;
	_step = 1;
    }
}

void
SliderCmd::setPercent(float percent)
{
    setValueInternal(_min + percent * (_max - _min));
}

void
SliderCmd::setValueInternal(float val)
{
    if (_step != 0)
	_val = _min + std::floor((val - _min)/_step)*_step;
    else
	_val = val;
    std::ostringstream	s;
    s << std::setw(5) << std::setfill(' ') << std::setprecision(4) << _val.f();
    XtVaSetValues(_text, XtNlabel, s.str().c_str(), nullptr);
}

}
}
