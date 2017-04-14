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
*  class MyOglCanvasPaneBase<D>						*
************************************************************************/
template <class D>
class MyOglCanvasPaneBase : public CanvasPane
{
  public:
    enum DrawMode	{Texture, Dot, Polygon, Mesh};

  public:
    MyOglCanvasPaneBase(Window&		parentWin,
			size_t		width,
			size_t		height,
			const Image<D>&	disparityMap)	;

    const OglDC&	dc()				const	{return _dc;}
    
    void		initialize(const Matrix34d& Pl,
				   const Matrix34d& Pr,
				   double scale=1.0)		;
    void		setDrawMode(DrawMode)			;
    void		setCursor(int u, int v, D d)		;
    void		setDistance(double distance)		;
    void		setParallax(double parallax)		;
    void		resetSwingView()			{_tick = 0;}
    void		swingView()				;
    void		resize(size_t w, size_t h)		;
    template <class T>
    Image<T>		getImage()			const	;
    
  protected:
    virtual void	initializeGraphics()			;

  protected:
    OglDC		_dc;
    const Image<D>&	_disparityMap;
    DrawThreeD		_draw;
    DrawMode		_drawMode;
    int			_tick;
    double		_parallax;
};

template <class D>
MyOglCanvasPaneBase<D>::MyOglCanvasPaneBase(Window&		parentWin,
					    size_t		width,
					    size_t		height,
					    const Image<D>&	disparityMap)
    :CanvasPane(parentWin, width, height), _draw(), _dc(*this),
     _disparityMap(disparityMap), _drawMode(Texture), _tick(0), _parallax(-1.0)
{
}

template <class D> void
MyOglCanvasPaneBase<D>::initialize(const Matrix34d& Pl,
				   const Matrix34d& Pr, double scale)
{
    typedef Camera<IntrinsicBase<double> >	camera_type;
    
    _draw.initialize(Pl, Pr);

    camera_type	camera(Pl);
    _dc.setInternal(camera.u0()[0] * scale, camera.u0()[1] * scale,
		    camera.k() * scale, camera.k() * scale, 0.01)
       .setExternal(camera.t(), camera.Rt());
    _dc << distance(1300.0);

    glPushMatrix();
}

template <class D> inline void
MyOglCanvasPaneBase<D>::setDrawMode(DrawMode drawMode)
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
    
template <class D> inline void
MyOglCanvasPaneBase<D>::setCursor(int u, int v, D d)
{
    _draw.setCursor(u, v, d);
}
    
template <class D> inline void
MyOglCanvasPaneBase<D>::setDistance(double d)
{
    _dc << distance(d);
}

template <class D> inline void
MyOglCanvasPaneBase<D>::setParallax(double parallax)
{
    glDrawBuffer(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _parallax = parallax;
}

template <class D> void
MyOglCanvasPaneBase<D>::swingView()
{
    const double	RAD = M_PI / 180.0;
    const double	magnitudeX = 45*RAD, magnitudeY = 45*RAD;
    const int		periodX = 400, periodY = 600;
    
    double		angleX = magnitudeX * sin(2.0*M_PI*_tick / periodX),
			angleY = magnitudeY * sin(2.0*M_PI*_tick / periodY);
    glPopMatrix();
    glPushMatrix();
    _dc << TU::v::axis(DC3::X) << TU::v::rotate(angleX)
	<< TU::v::axis(DC3::Y) << TU::v::rotate(angleY);
    ++_tick;
}

template <class D> inline void
MyOglCanvasPaneBase<D>::resize(size_t w, size_t h)
{
    _dc.setSize(w, h, _dc.zoom());
}

template <class D> template <class T> inline Image<T>
MyOglCanvasPaneBase<D>::getImage() const
{
    return _dc.getImage<T>();
}

template <class D> void
MyOglCanvasPaneBase<D>::initializeGraphics()
{
  //    glClearColor(0.0, 0.1, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    glFrontFace(GL_CW);
    glCullFace(GL_FRONT);
    glEnable(GL_CULL_FACE);

    glEnable(GL_COLOR_MATERIAL);
    glDisable(GL_AUTO_NORMAL);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_FLAT);

    GLfloat	position[] = {1.0, 1.0, -1.0, 0.0};
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glEnable(GL_LIGHT0);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_DECAL);

    setDrawMode(_drawMode);
}

/************************************************************************
*  class MyOglCanvasPane<D, T>						*
************************************************************************/
template <class D, class T>
class MyOglCanvasPane : public MyOglCanvasPaneBase<D>
{
  private:
    typedef MyOglCanvasPaneBase<D>	super;
    
  public:
			MyOglCanvasPane(
			    Window&		parentWin,
			    size_t		width,
			    size_t		height,
			    const Image<D>&	disparityMap,
			    const Image<T>&	textureImage,
			    const Warp*		warp)		;
    virtual		~MyOglCanvasPane()			;
    
    virtual void	repaintUnderlay()			;
    
  private:
    using		super::_dc;
    using		super::_disparityMap;
    using		super::_draw;
    using		super::_drawMode;
    using		super::_parallax;
    
    const Image<T>&	_textureImage;
    const Warp* const	_warp;
    GLuint		_list;
};

template <class D, class T> inline
MyOglCanvasPane<D, T>::MyOglCanvasPane(Window&		parentWin,
				       size_t		width,
				       size_t		height,
				       const Image<D>&	disparityMap,
				       const Image<T>&	textureImage,
				       const Warp*	warp)
    :MyOglCanvasPaneBase<D>(parentWin, width, height, disparityMap),
     _textureImage(textureImage), _warp(warp), _list(glGenLists(1))
{
}

template <class D, class T>
MyOglCanvasPane<D, T>::~MyOglCanvasPane()
{
    glDeleteLists(_list, 1);
}

template <class D, class T> void
MyOglCanvasPane<D, T>::repaintUnderlay()
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
      case super::Texture:
	if (_warp)
	    _draw.draw(_disparityMap, _textureImage, *_warp);
	else
	    _draw.draw(_disparityMap, _textureImage);
	break;
      case super::Polygon:
	_draw.template draw<DrawThreeD::N3F_V3F>(_disparityMap);
	break;
      case super::Mesh:
	_draw.template draw<DrawThreeD::V3F>(_disparityMap);
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

