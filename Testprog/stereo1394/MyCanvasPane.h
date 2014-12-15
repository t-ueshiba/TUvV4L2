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
 *  $Id: MyCanvasPane.h 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#include "TU/v/ShmDC.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCanvasPaneBase						*
************************************************************************/
class MyCanvasPaneBase : public CanvasPane
{
  public:
    MyCanvasPaneBase(Window& parentWin, size_t width, size_t height)
	:CanvasPane(parentWin, width, height),
	 _dc(*this)					{_dc << cross;}
    
    void		resize(size_t w, size_t h)			;
    void		setZoom(size_t mul, size_t div)			;
    void		setSaturation(size_t saturation)			;
    void		drawEpipolarLine(int v)				;
    void		eraseEpipolarLine(int v)			;
    void		drawEpipolarLineV(int u)			;
    void		eraseEpipolarLineV(int u)			;
    void		drawPoint(int u, int v)				;
    void		erasePoint(int u, int v)			;
    void		clearOverlay()					;
    virtual void	callback(CmdId id, CmdVal val)			;
    
  protected:
#if defined(USE_CANVAS_PANE_DC)
    CanvasPaneDC	_dc;
#else
    ShmDC		_dc;
#endif
};

inline void
MyCanvasPaneBase::resize(size_t w, size_t h)
{
    _dc.setSize(w, h, _dc.mul(), _dc.div());
}

inline void
MyCanvasPaneBase::setZoom(size_t mul, size_t div)
{
    _dc.setSize(_dc.width(), _dc.height(), mul, div);
}

inline void
MyCanvasPaneBase::setSaturation(size_t saturation)
{
    _dc.setSaturation(saturation);
}

/************************************************************************
*  class MyCanvasPane<T>						*
************************************************************************/
template <class T>
class MyCanvasPane : public MyCanvasPaneBase
{
  public:
    MyCanvasPane(Window& parentWin, size_t width, size_t height,
		 const Image<T>& image)
	:MyCanvasPaneBase(parentWin, width, height), _image(image)	{}

    virtual void	repaintUnderlay()				;
    
  private:
    const Image<T>&	_image;
};

template <class T> void
MyCanvasPane<T>::repaintUnderlay()
{
    _dc << clear << _image;
}

}
}
