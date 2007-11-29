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
 *  $Id: CanvasPaneDC3.cc,v 1.4 2007-11-29 07:06:06 ueshiba Exp $
 */
#include "TU/v/CanvasPaneDC3.h"
#include <X11/keysym.h>

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPaneDC							*
************************************************************************/
/*
 *  Public member functions
 */
CanvasPaneDC3::CanvasPaneDC3(CanvasPane& parentCanvasPane, u_int w, u_int h)
    :CanvasPaneDC(parentCanvasPane, w, h),
     DC3(DC3::X, 128.0)
{
}

CanvasPaneDC3::~CanvasPaneDC3()
{
}

void
CanvasPaneDC3::callback(CmdId id, CmdVal val)
{
    if (id == Id_KeyPress)
	switch (val)
	{
	  case 'n':
	    *this << TU::v::axis(Z) << TU::v::rotate( 5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'm':
	    *this << TU::v::axis(Z) << TU::v::rotate(-5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'h':
	    *this << TU::v::axis(Y) << TU::v::rotate( 5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'j':
	    *this << TU::v::axis(X) << TU::v::rotate( 5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'k':
	    *this << TU::v::axis(X) << TU::v::rotate(-5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;
	  case 'l':
	    *this << TU::v::axis(Y) << TU::v::rotate(-5 * M_PI / 180.0)
		  << TU::v::repaint;
	    break;;

	  case 'N':
	    *this << TU::v::axis(Z) << TU::v::translate(-0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'M':
	    *this << TU::v::axis(Z) << TU::v::translate( 0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'H':
	    *this << TU::v::axis(X) << TU::v::translate( 0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'J':
	    *this << TU::v::axis(Y) << TU::v::translate(-0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'K':
	    *this << TU::v::axis(Y) << TU::v::translate( 0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	  case 'L':
	    *this << TU::v::axis(X) << TU::v::translate(-0.05 * getDistance())
		  << TU::v::repaint;
	    break;;
	}
    else
	CanvasPaneDC::callback(id, val);
}

/*
 *  Protected member functions
 */
void
CanvasPaneDC3::initializeGraphics()
{
  // Set initial internal and external parameters.
    setInternal(width() / 2, height() / 2, 800.0, 800.0, 1.0, 1000.0);
    Matrix<double>	Rt(3, 3);
    Rt[0][0] = Rt[2][1] = 1.0;
    Rt[1][2] = -1.0;
    setExternal(Vector<double>(3), Rt);
}

}
}
