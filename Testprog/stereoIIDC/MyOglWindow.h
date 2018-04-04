/*
 *  $Id: MyOglWindow.h 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include "TU/v/CmdWindow.h"
#include "MyOglCanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyOglWindow							*
************************************************************************/
template <class T>
class MyOglWindow : public CmdWindow
{
  public:
    MyOglWindow(Window&			parentWin,
		const XVisualInfo*	vinfo,
		size_t			width,
		size_t			height,
		const Image<float>&	dispalrityMap,
		const Image<T>&		textureImage,
		const Warp*		warp)		;

    const MyOglCanvasPane<T>&	canvas()	const	{return _canvas;}
    MyOglCanvasPane<T>&		canvas()		{return _canvas;}
    
  private:
    MyOglCanvasPane<T>		_canvas;
};

template <class T> inline
MyOglWindow<T>::MyOglWindow(Window&		parentWin,
			    const XVisualInfo*	vinfo,
			    size_t		width,
			    size_t		height,
			    const Image<float>&	disparityMap,
			    const Image<T>&	textureImage,
			    const Warp*		warp)
    :CmdWindow(parentWin, "3D canvas",
	       vinfo, Colormap::RGBColor, 256, 0, 0, true),
     _canvas(*this, width, height, disparityMap, textureImage, warp)
{
    _canvas.place(0, 0, 1, 1);
    show();
    _canvas.dc().grabKeyboard();
}

}
}
