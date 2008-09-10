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
 *  $Id: CanvasPaneDC3.h,v 1.5 2008-09-10 05:12:00 ueshiba Exp $  
 */
#ifndef __TUvCanvasPaneDC3_h
#define __TUvCanvasPaneDC3_h

#include "TU/v/CanvasPaneDC.h"
#include "TU/v/DC3.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPaneDC3							*
************************************************************************/
class CanvasPaneDC3 : virtual public CanvasPaneDC, public DC3
{
  public:
    CanvasPaneDC3(CanvasPane& parentCanvasPane,
		  u_int width=0, u_int height=0)			;
    virtual		~CanvasPaneDC3()				;
    virtual void	callback(CmdId id, CmdVal val)			;

  protected:
    virtual void	initializeGraphics()				;
};

}
}
#endif	// !__TUvCanvasPaneDC3_h
