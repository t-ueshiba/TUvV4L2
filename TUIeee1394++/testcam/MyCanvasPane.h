/*
 *  $Id: MyCanvasPane.h,v 1.1 2009-07-28 00:00:48 ueshiba Exp $
 */
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#if defined(UseShmDC)
#  include "TU/v/ShmDC.h"
#elif defined(UseXvDC)
#  include "TU/v/XvDC.h"
#endif

namespace TU
{
#ifdef MONO_IMAGE
typedef u_char		PixelType;
#else
typedef RGBA		PixelType;
#endif

namespace v
{
/************************************************************************
*  class MyCanvasPane							*
************************************************************************/
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, u_int width, u_int height,
		 const Image<PixelType>& image)
	:CanvasPane(parentWin, width, height),
	_dc(*this, 640, 480), _image(image)				{}
    
    void		resize(u_int w, u_int h)			;
    virtual void	repaintUnderlay()				;
    
  private:
#if defined(UseShmDC)
    ShmDC			_dc;
#elif defined(UseXvDC)
    XvDC			_dc;
#else
    CanvasPaneDC		_dc;
#endif
    const Image<PixelType>&	_image;
};

inline void
MyCanvasPane::resize(u_int w, u_int h)
{
    _dc.setSize(w, h, _dc.mul(), _dc.div());
}

}
}
