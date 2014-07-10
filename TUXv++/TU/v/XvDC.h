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
 *  $Id$
 */
#ifndef __TU_V_XVDC_H
#define __TU_V_XVDC_H

#include "TU/v/ShmDC.h"
#include <X11/extensions/Xvlib.h>

namespace TU
{
namespace v
{
/************************************************************************
*  class XvDC								*
************************************************************************/
class XvDC : public ShmDC, public List<XvDC>::Node
{
  public:
    XvDC(CanvasPane& parentCanvasPane,
	 u_int width=0, u_int height=0, u_int mul=1, u_int div=1)	;
    virtual		~XvDC()						;

    using		ShmDC::operator <<;
    virtual DC&		operator <<(const Point2<int>& p)		;
    virtual DC&		operator <<(const LineP2f& l)			;
    virtual DC&		operator <<(const LineP2d& l)			;
    virtual DC&		operator <<(const Image<u_char>& image)		;
    virtual DC&		operator <<(const Image<s_char>& image)		;
    virtual DC&		operator <<(const Image<short>&  image)		;
    virtual DC&		operator <<(const Image<BGR>&    image)		;
    virtual DC&		operator <<(const Image<ABGR>&   image)		;
    virtual DC&		operator <<(const Image<RGB>&    image)		;
    virtual DC&		operator <<(const Image<RGBA>&   image)		;
    virtual DC&		operator <<(const Image<YUV444>& image)		;
    virtual DC&		operator <<(const Image<YUV422>& image)		;
    virtual DC&		operator <<(const Image<YUYV422>& image)	;
    virtual DC&		operator <<(const Image<YUV411>& image)		;

  protected:
    virtual void	destroyShmImage()				;
    
  private:
    enum FormatId
    {
	I420 = 0x30323449,
	RV15 = 0x35315652,	// RGB 15bits:
	RV16 = 0x36315652,	// RGB 16bits:
	YV12 = 0x32315659,	// 4:2:0 Planar mode:	Y + V + U
	YUY2 = 0x32595559,	// 4:2:2 Packed mode:	Y0 + U + Y1 + V
	UYVY = 0x59565955,	// 4:2:2 Packed mode:	U + Y0 + V + Y1
        BGRA = 0x3		// RGBA 32bits:		B + G + R + A
    };

    template <class S>
    void		createXvImage(const Image<S>& image)		;
    XvPortID		grabPort(XvPortID base_id, u_long num_ports)	;
    
    XvPortID		_port;
    XvImage*		_xvimage;

    static List<XvDC>	_vXvDCList;
};

}
}
#endif	// !__TU_V_XVDC_H
