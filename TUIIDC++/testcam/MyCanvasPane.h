/*
 *  $Id: MyCanvasPane.h,v 1.2 2010-12-28 11:47:48 ueshiba Exp $
 */
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#if defined(USE_SHMDC)
#  include "TU/v/ShmDC.h"
#elif defined(USE_XVDC)
#  include "TU/v/XvDC.h"
#endif

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCanvasPane<PIXEL>						*
************************************************************************/
template <class PIXEL>
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, u_int width, u_int height,
		 const Image<PIXEL>& image)
	:CanvasPane(parentWin, width, height),
	 _dc(*this, image.width(), image.height()), _image(image)	{}
    MyCanvasPane(Window& parentWin, const Image<PIXEL>& image,
		 u_int w, u_int h, float zoom)
	:CanvasPane(parentWin, u_int(w * zoom), u_int(h * zoom)),
	 _dc(*this, w, h, zoom), _image(image)				{}
    
    void		resize()					;
    virtual void	repaintUnderlay()				;
    
  private:
#if defined(USE_SHMDC)
    ShmDC		_dc;
#elif defined(USE_XVDC)
    XvDC		_dc;
#else
    CanvasPaneDC	_dc;
#endif
    const Image<PIXEL>&	_image;
};

template <class PIXEL> inline void
MyCanvasPane<PIXEL>::resize()
{
    _dc.setSize(_image.width(), _image.height(), _dc.zoom());
}

template <class PIXEL> void
MyCanvasPane<PIXEL>::repaintUnderlay()
{
    _dc << _image;
}

}
}
