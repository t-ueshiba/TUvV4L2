/*
 *  $Id: SliderCmd.cc,v 1.2 2002-07-25 02:38:12 ueshiba Exp $
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
				       (cmd.size != 0 ?	cmd.size : 100),
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
    XtVaSetValues(_text, XtNlabel, s.str().data(), NULL);
}

}
}
