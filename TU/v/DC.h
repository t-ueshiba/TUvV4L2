/*
 *  $Id$  
 */
#ifndef TU_V_DC_H
#define TU_V_DC_H

#include "TU/Geometry++.h"
#include "TU/Image++.h"
#include "TU/Manip.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class DC:		coorindate system for drawing			*
************************************************************************/
class DC
{
  public:
    enum Layer		{UNDERLAY, OVERLAY};
    enum PointStyle	{DOT, CROSS, CIRCLE};
    
  public:
    DC(size_t width, size_t height, float zoom)
	:_width(width), _height(height), _zoom(zoom),
	 _offset({0, 0}), _layer(UNDERLAY), _pointStyle(DOT)		{}
    virtual		~DC()						;
    
    size_t		width()				const	{return _width;}
    size_t		height()			const	{return _height;}
    float		zoom()				const	{return _zoom;}
    const Point2<int>&	offset()			const	{return _offset;}

    virtual DC&		setSize(size_t width, size_t height, float zoom);
	    DC&		setZoom(float zoom)				;
    virtual DC&		setOffset(int u0, int v0)			;
    virtual DC&		setLayer(Layer layer)				;
	    DC&		setPointStyle(PointStyle pointStyle)		;
    virtual DC&		setThickness(size_t thickness)			= 0;
    virtual DC&		setForeground(const BGR& fg)			= 0;
    virtual DC&		setBackground(const BGR& bg)			= 0;
    virtual DC&		setForeground(size_t index)			= 0;
    virtual DC&		setBackground(size_t index)			= 0;
    virtual DC&		setSaturation(size_t saturation)		= 0;
    virtual DC&		setSaturationF(float saturation)		= 0;
    
    virtual DC&		clear()						= 0;
	    DC&		repaint()					;
	    DC&		repaintAll()					;
    virtual DC&		sync()						= 0;
    
    virtual DC&		operator <<(const Point2<int>& p)		= 0;
    template <class T>
	    DC&		operator <<(const Point2<T>& p)			;
    virtual DC&		operator <<(const LineP2f& l)			= 0;
    virtual DC&		operator <<(const LineP2d& l)			= 0;
    virtual DC&		operator <<(const Image<u_char>& image)		= 0;
    virtual DC&		operator <<(const Image<s_char>& image)		= 0;
    virtual DC&		operator <<(const Image<short>&  image)		= 0;
    virtual DC&		operator <<(const Image<float>&  image)		= 0;
    virtual DC&		operator <<(const Image<BGR>&    image)		= 0;
    virtual DC&		operator <<(const Image<ABGR>&   image)		= 0;
    virtual DC&		operator <<(const Image<BGRA>&   image)		= 0;
    virtual DC&		operator <<(const Image<RGB>&    image)		= 0;
    virtual DC&		operator <<(const Image<RGBA>&   image)		= 0;
    virtual DC&		operator <<(const Image<ARGB>&   image)		= 0;
    virtual DC&		operator <<(const Image<YUV444>& image)		= 0;
    virtual DC&		operator <<(const Image<YUV422>& image)		= 0;
    virtual DC&		operator <<(const Image<YUYV422>& image)	= 0;
    virtual DC&		operator <<(const Image<YUV411>& image)		= 0;
    virtual DC&		drawLine(const Point2<int>& p,
				 const Point2<int>& q)			= 0;
    template <class T>
	    DC&		drawLine(const Point2<T>& p,
				 const Point2<T>& q)			;
    virtual DC&		draw(const char* s, int u, int v)		= 0;

    int			log2devR(int r)				const	;
    int			log2devU(int u)				const	;
    int			log2devV(int v)				const	;
    Point2<int>		log2dev(const Point2<int>& p)		const	;
    int			dev2logR(int r)				const	;
    int			dev2logU(int u)				const	;
    int			dev2logV(int v)				const	;
    Point2<int>		dev2log(const Point2<int>& p)		const	;

  protected:
    enum		{PRADIUS = 5};		// radius of marker points
	    
    int			hasScale()		const	{return _zoom != 1.0f;}
    size_t		deviceWidth()		const	;
    size_t		deviceHeight()		const	;

    Layer		getLayer()		const	;
    PointStyle		getPointStyle()		const	;
    virtual size_t	getThickness()		const	= 0;
    
    virtual DC&		repaintUnderlay()		= 0;
    virtual DC&		repaintOverlay()		= 0;
    
  private:
    size_t		_width;		// logical width  of DC
    size_t		_height;	// logical height of DC
    float		_zoom;		// zooming factor 
    Point2<int>		_offset;	// logical coordinates of the
					// temporary offset
    Layer		_layer;		// flag indicating underlay/overlay
    PointStyle		_pointStyle;	// drawing style for points
};

inline DC&
DC::setZoom(float zoom)
{
    return setSize(_width, _height, zoom);
}
    
template <class T> inline DC&
DC::operator <<(const Point2<T>& p)
{
    return *this << Point2<int>(p);
}

template <class T> inline DC&
DC::drawLine(const Point2<T>& p, const Point2<T>& q)
{
    return drawLine(Point2<int>(p), Point2<int>(q));
}

inline int
DC::log2devR(int r)	const	{return int(r * _zoom);}

inline int
DC::log2devU(int u)	const	{return log2devR(u + _offset[0]) + 0.5f*_zoom;}

inline int
DC::log2devV(int v)	const	{return log2devR(v + _offset[1]) + 0.5f*_zoom;}

inline Point2<int>
DC::log2dev(const Point2<int>& p) const
{
    if (hasScale())
	return Point2<int>(log2devU(p[0]), log2devV(p[1]));
    else
	return Point2<int>(p[0] + _offset[0], p[1] + _offset[1]);
}

inline int
DC::dev2logR(int r)	const	{return int(r / _zoom);}

inline int
DC::dev2logU(int u)	const	{return dev2logR(u);}

inline int
DC::dev2logV(int v)	const	{return dev2logR(v);}

inline Point2<int>
DC::dev2log(const Point2<int>& p) const
{
    if (hasScale())
	return Point2<int>(dev2logU(p[0]), dev2logV(p[1]));
    else
	return Point2<int>(p[0] - _offset[0], p[1] - _offset[1]);
}

inline size_t
DC::deviceWidth()	const	{return log2devR(_width);}

inline size_t
DC::deviceHeight()	const	{return log2devR(_height);}

inline DC&
DC::setPointStyle(PointStyle pointStyle)
{
    _pointStyle = pointStyle;
    return *this;
}

inline DC::Layer	DC::getLayer()		const	{return _layer;}
inline DC::PointStyle	DC::getPointStyle()	const	{return _pointStyle;}

/************************************************************************
*  Manipulators								*
************************************************************************/
extern DC&			x0_125(DC&)		;
extern DC&			x0_25(DC&)		;
extern DC&			x0_5(DC&)		;
extern DC&			x1(DC&)			;
extern DC&			x1_5(DC&)		;
extern DC&			x2(DC&)			;
extern DC&			x4(DC&)			;
extern DC&			x8(DC&)			;
extern DC&			underlay(DC&)		;
extern DC&			overlay(DC&)		;
extern DC&			dot(DC&)		;
extern DC&			cross(DC&)		;
extern DC&			circle(DC&)		;
extern DC&			clear(DC&)		;
extern DC&			repaint(DC&)		;
extern DC&			repaintAll(DC&)		;
extern DC&			sync(DC&)		;
extern OManip1<DC, const BGR&>	foreground(const BGR&)	;
extern OManip1<DC, const BGR&>	background(const BGR&)	;
extern OManip1<DC, size_t>	foreground(size_t)	;
extern OManip1<DC, size_t>	background(size_t)	;
extern OManip1<DC, size_t>	thickness(size_t)	;
extern OManip1<DC, size_t>	saturation(size_t)	;
extern OManip1<DC, float>	saturationF(float)	;
extern OManip2<DC, int, int>	offset(int, int)	;

template <class S> inline S&
operator <<(S& dc, DC& (*f)(DC&))
{
    (*f)(dc);
    return dc;
}

}
}
#endif	// !TU_V_DC_H
