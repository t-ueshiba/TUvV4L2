/*
 *  $Id: MyCanvasPane.h,v 1.2 2010-12-28 11:47:48 ueshiba Exp $
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
	 _dc(*this, image.width(), image.height()), _image(image)	{}
    
    void		resize()					;
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
MyCanvasPane::resize()
{
    _dc.setSize(_image.width(), _image.height(), _dc.mul(), _dc.div());
}

}
}
