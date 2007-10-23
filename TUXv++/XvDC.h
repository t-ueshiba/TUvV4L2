/*
 *  $Id: XvDC.h,v 1.3 2007-10-23 02:27:07 ueshiba Exp $
 */
#ifndef __TUvXvDC_h
#define __TUvXvDC_h

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
    XvDC(CanvasPane& parentCanvasPane, u_int width=0, u_int height=0)	;
    virtual		~XvDC()						;
    
    virtual DC&		operator <<(const Point2<int>& p)		;
    virtual DC&		operator <<(const LineP2d& l)			;
    virtual DC&		operator <<(const Image<u_char>& image)		;
    virtual DC&		operator <<(const Image<short>&  image)		;
    virtual DC&		operator <<(const Image<BGR>&  image)		;
    virtual DC&		operator <<(const Image<ABGR>& image)		;
    virtual DC&		operator <<(const Image<YUV444>& image)		;
    virtual DC&		operator <<(const Image<YUV422>& image)		;
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
#endif	// !__TUvXvDC_h
