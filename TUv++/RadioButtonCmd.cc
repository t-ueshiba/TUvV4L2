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
 *  $Id: RadioButtonCmd.cc,v 1.6 2008-12-24 01:01:57 ueshiba Exp $  
 */
#include "TU/v/Bitmap.h"
#include "RadioButtonCmd_.h"
#include <X11/Xaw3d/Command.h>
#include <X11/Xaw3d/ThreeD.h>
#include "vGridbox_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  bitmaps for RadioButtonCmd						*
************************************************************************/
static u_char	onBits[] =
		{
		    11, 11,
		    0x70, 0x00, 0x8c, 0x01, 0x72, 0x02,
		    0xfa, 0x02, 0xfd, 0x05, 0xfd, 0x05,
		    0xfd, 0x05, 0xfa, 0x02, 0x72, 0x02,
		    0x8c, 0x01, 0x70, 0x00
		};
static Bitmap*	onBitmap = 0;

static u_char	offBits[] =
		{
		    11, 11,
		    0x70, 0x00, 0x8c, 0x01, 0x02, 0x02,
		    0x02, 0x02, 0x01, 0x04, 0x01, 0x04,
		    0x01, 0x04, 0x02, 0x02, 0x02, 0x02,
		    0x8c, 0x01, 0x70, 0x00
		};
static Bitmap*	offBitmap = 0;

/************************************************************************
*  class RadioButtonCmd							*
************************************************************************/
u_int		RadioButtonCmd::_nitems = 0;

RadioButtonCmd::RadioButtonCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvRadioButtonCmd", cmd),
     _button(XtVaCreateManagedWidget("TUvRadioButtonCmd-button",
				     commandWidgetClass,
				     _widget,
				     XtNbackground,
				     parent().widget().background(),
#ifdef XtNrelief
				     XtNrelief,			XtReliefNone,
#else
				     XtNshadowWidth,		0,
#endif
				     XtNborderWidth,		0,
				     NULL)),
     _label(XtVaCreateManagedWidget("TUvRadioButtonCmd-label",
				    labelWidgetClass,
				    _widget,
				    XtNbackground,
				    parent().widget().background(),
				    XtNlabel,			cmd.title,
				    XtNinternalHeight,		3,
				    XtNinternalWidth,		0,
				    XtNborderWidth,		0,
				    XtNhighlightThickness,	0,
				    XtNgridx,			1,
				    NULL))
{
    if (_nitems++ == 0)
    {
	onBitmap  = new Bitmap(window().colormap(), onBits);
	offBitmap = new Bitmap(window().colormap(), offBits);
    }
	
    setValue(cmd.val);
    setDefaultCallback(_button);
}

RadioButtonCmd::~RadioButtonCmd()
{
    if (--_nitems == 0)
    {
	delete offBitmap;
	delete onBitmap;
    }
}

const Object::Widget&
RadioButtonCmd::widget() const
{
    return _widget;
}

CmdVal
RadioButtonCmd::getValue() const
{
    Pixmap	bitmap;
    XtVaGetValues(_button, XtNbitmap, &bitmap, NULL);
    return (bitmap == onBitmap->xpixmap() ? 1 : 0);
}

void
RadioButtonCmd::setValue(CmdVal val)
{
    XtVaSetValues(_button,
		  XtNbitmap,	(val != 0 ?
				 onBitmap->xpixmap() : offBitmap->xpixmap()),
		  NULL);
}

}
}
