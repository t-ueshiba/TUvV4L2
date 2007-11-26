/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，使用，第三者へ開示する
 *  等の著作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: CanvasPane.cc,v 1.3 2007-11-26 08:11:50 ueshiba Exp $
 */
#include "TU/v/CanvasPane.h"
#include "vViewport_.h"
#include "vGridbox_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPane							*
************************************************************************/
CanvasPane::CanvasPane(Window& parentWin, u_int devWidth, u_int devHeight)
    :Pane(parentWin),
     _widget(XtVaCreateManagedWidget("TUvCanvasPane",
				     vViewportWidgetClass,  // widget class 
				     parent().widget(),     // parent widget
				     XtNallowHoriz,	TRUE,
				     XtNallowVert,	TRUE,
				     XtNwidth,	(devWidth != 0 ? devWidth :
						 parent().widget().width()),
				     XtNheight,	(devHeight != 0 ? devHeight :
						 parent().widget().height()),
				     XtNuseBottom,	TRUE,
				     XtNuseRight,	TRUE,

				   // Expand/shrink according to the width/
				   // height of Paned.
				     XtNweightx,	1,
				     XtNweighty,	1,
				     NULL))
{
}

CanvasPane::~CanvasPane()
{
}

const Object::Widget&
CanvasPane::widget() const
{
    return _widget;
}

CanvasPane&
CanvasPane::canvasPane()
{
    return *this;
}

void
CanvasPane::initializeGraphics()
{
}

void
CanvasPane::repaintUnderlay(int /* u */, int /* v */,
			    int /* width */, int /* height */)
{
}

void
CanvasPane::repaintOverlay(int /* u */, int /* v */,
			   int /* width */, int /* height */)
{
}

void
CanvasPane::moveDC(int u, int v)
{
    vViewportSetCoordinates(_widget,
			    u - _widget.width()/2, v - _widget.height()/2);
}

}
}
