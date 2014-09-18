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
 *  $Id: MyOglCanvasPane.h 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include "TU/v/CanvasPane.h"
#include "TU/v/OglDC.h"
#include "DrawThreeD.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyOglCanvasPaneBase						*
************************************************************************/
class MyOglCanvasPaneBase : public CanvasPane
{
  public:
    enum DrawMode	{Texture, Dot, Polygon, Mesh};

  public:
    MyOglCanvasPaneBase(Window&			parentWin,
			size_t			width,
			size_t			height,
			const Image<float>&	disparityMap)	;

    const OglDC&	dc()				const	{return _dc;}
    
    void		initialize(const Matrix34d& Pl,
				   const Matrix34d& Pr,
				   double scale=1.0)		;
    void		setDrawMode(DrawMode)			;
    void		setDistance(double distance)		;
    void		resetSwingView()			{_tick = 0;}
    void		swingView()				;
    void		setParallax(double parallax)		;
    void		setCursor(int u, int v, float d)	;
    template <class T>
    Image<T>		getImage()			const	;
    
    void		resize(size_t w, size_t h)		;
    
  protected:
    virtual void	initializeGraphics()			;

  protected:
    OglDC		_dc;
    const Image<float>&	_disparityMap;
    DrawThreeD		_draw;
    DrawMode		_drawMode;
    int			_tick;
    double		_parallax;
};

inline void
MyOglCanvasPaneBase::setDrawMode(DrawMode drawMode)
{
    _drawMode = drawMode;
    switch (_drawMode)
    {
      case Texture:
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	break;
      case Polygon:
	glEnable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	break;
      case Mesh:
	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	break;
    }
}
    
inline void
MyOglCanvasPaneBase::setDistance(double d)
{
    _dc << distance(d);
}

inline void
MyOglCanvasPaneBase::setCursor(int u, int v, float d)
{
    _draw.setCursor(u, v, d);
}
    
inline void
MyOglCanvasPaneBase::setParallax(double parallax)
{
    glDrawBuffer(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _parallax = parallax;
}

inline void
MyOglCanvasPaneBase::resize(size_t w, size_t h)
{
    _dc.setSize(w, h, _dc.mul(), _dc.div());
}

template <class T> inline Image<T>
MyOglCanvasPaneBase::getImage() const
{
    return _dc.getImage<T>();
}

/************************************************************************
*  class MyOglCanvasPane<T>						*
************************************************************************/
template <class T>
class MyOglCanvasPane : public MyOglCanvasPaneBase
{
  public:
			MyOglCanvasPane(
			    Window&		parentWin,
			    size_t		width,
			    size_t		height,
			    const Image<float>&	disparityMap,
			    const Image<T>&	textureImage,
			    const Warp*		warp)		;
    virtual		~MyOglCanvasPane()			;
    
    virtual void	repaintUnderlay()			;
    
  private:
    const Image<T>&	_textureImage;
    const Warp* const	_warp;
    GLuint		_list;
};

template <class T> inline
MyOglCanvasPane<T>::MyOglCanvasPane(Window&		parentWin,
				    size_t		width,
				    size_t		height,
				    const Image<float>&	disparityMap,
				    const Image<T>&	textureImage,
				    const Warp*		warp)
    :MyOglCanvasPaneBase(parentWin, width, height, disparityMap),
     _textureImage(textureImage), _warp(warp), _list(glGenLists(1))
{
}

template <class T>
MyOglCanvasPane<T>::~MyOglCanvasPane()
{
    glDeleteLists(_list, 1);
}

template <class T> void
MyOglCanvasPane<T>::repaintUnderlay()
{
    GLuint	list;
    
    if (_parallax > 0.0)
    {
	glPushMatrix();	// Keep the current model-view transformation.

	_dc << v::axis(DC3::X) << v::translate(-_parallax / 2.0);
	glDrawBuffer(GL_BACK_LEFT);
	glNewList(_list, GL_COMPILE_AND_EXECUTE);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    switch (_drawMode)
    {
      case Texture:
	if (_warp)
	    _draw.draw(_disparityMap, _textureImage, *_warp);
	else
	    _draw.draw(_disparityMap, _textureImage);
	break;
      case Polygon:
	_draw.draw<DrawThreeD::N3F_V3F>(_disparityMap);
	break;
      case Mesh:
	_draw.draw<DrawThreeD::V3F>(_disparityMap);
	break;
    }

    if (_parallax > 0.0)
    {
	glEndList();

	_dc << v::translate(_parallax);
	glDrawBuffer(GL_BACK_RIGHT);
	glCallList(_list);
	
	glPopMatrix();	// Restore the original model-view transformation.
    }

    _dc.swapBuffers();
}

}
}

