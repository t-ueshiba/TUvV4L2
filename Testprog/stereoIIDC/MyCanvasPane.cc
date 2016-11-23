/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
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
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: MyCanvasPane.cc 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include "MyCanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCanvasPaneBase						*
************************************************************************/
void
MyCanvasPaneBase::drawEpipolarLine(int v)
{
    LineP2d	l;
    l[0] = 0.0;
    l[1] = 1.0;
    l[2] = -v;
#if defined(USE_OVERLAY)
    _dc << overlay << foreground(1) << l << underlay;
#else
    _dc	<< foreground(BGR(0, 255, 0)) << l;
#endif
}

void
MyCanvasPaneBase::eraseEpipolarLine(int v)
{
    LineP2d	l;
    l[0] = 0.0;
    l[1] = 1.0;
    l[2] = -v;
#if defined(USE_OVERLAY)
    _dc << overlay << foreground(0) << l << underlay;
#else
    _dc	<< foreground(BGR(0, 0, 0)) << l;
#endif    
}

void
MyCanvasPaneBase::drawEpipolarLineV(int u)
{
    LineP2d	l;
    l[0] = 1.0;
    l[1] = 0.0;
    l[2] = -u;
#if defined(USE_OVERLAY)
    _dc << overlay << foreground(1) << l << underlay;
#else
    _dc << foreground(BGR(0, 255, 0)) << l;
#endif
}

void
MyCanvasPaneBase::eraseEpipolarLineV(int u)
{
    LineP2d	l;
    l[0] = 1.0;
    l[1] = 0.0;
    l[2] = -u;
#if defined(USE_OVERLAY)
    _dc << overlay << foreground(0) << l << underlay;
#else
    _dc << foreground(BGR(0, 0, 0)) << l;
#endif
}

void
MyCanvasPaneBase::drawPoint(int u, int v)
{
#if defined(USE_OVERLAY)
    _dc << overlay << foreground(1) << Point2<int>(u, v) << underlay;
#else
    _dc << foreground(BGR(0, 255, 0)) << Point2<int>(u, v);
#endif
}

void
MyCanvasPaneBase::erasePoint(int u, int v)
{
#if defined(USE_OVERLAY)
    _dc << overlay << foreground(0) << Point2<int>(u, v) << underlay;
#else
    _dc << foreground(BGR(0, 0, 0)) << Point2<int>(u, v);
#endif
}

void
MyCanvasPaneBase::clearOverlay()
{
    _dc << overlay << clear << underlay;
}

void
MyCanvasPaneBase::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case Id_MouseButton1Press:
      case Id_MouseButton1Drag:
      case Id_MouseButton1Release:
      case Id_MouseMove:
      {
	  CmdVal logicalPosition(_dc.dev2logU(val.u()), _dc.dev2logV(val.v()));
	  parent().callback(id, logicalPosition);
      }
	return;
    }

    parent().callback(id, val);
}

}
}
