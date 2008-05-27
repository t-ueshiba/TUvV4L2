/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 1997-2007.
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
 *  $Id: CanvasPane.h,v 1.5 2008-05-27 11:38:25 ueshiba Exp $
 */
#ifndef __TUvCanvasPane_h
#define __TUvCanvasPane_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPane							*
************************************************************************/
class CanvasPane : public Pane
{
  public:
    CanvasPane(Window& parentWin, u_int devWidth=0, u_int devHeight=0)	;
    virtual			~CanvasPane()				;

    virtual const Widget&	widget()			const	;

    virtual void		repaintUnderlay()			;
    virtual void		repaintOverlay()			;
    void			moveDC(int u, int v)			;
    
  protected:
    virtual CanvasPane&		canvasPane()				;
    virtual void		initializeGraphics()			;
    
  private:
  // allow access to initializeGraphics
    friend void		CBcanvasPaneDC(::Widget, XtPointer client_data,
				       XtPointer)			;

    const Widget	_widget;		// viewportWidget
};

}
}
#endif	// !__CanvasPane_h
