/*
 *  $Id: DC.h,v 1.6 2007-10-23 02:27:07 ueshiba Exp $
 */
#ifndef __TUvDC_h
#define __TUvDC_h

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
    DC(u_int width, u_int height)
	:_width(width), _height(height),
	 _mul(1), _div(1), _offset(0, 0),
	_layer(UNDERLAY), _pointStyle(DOT)		{}
    virtual		~DC()				;
    
    u_int		width()			const	{return _width;}
    u_int		height()		const	{return _height;}
    u_int		mul()			const	{return _mul;}
    u_int		div()			const	{return _div;}
    const Point2<int>&	offset()		const	{return _offset;}

    virtual DC&		setSize(u_int width, u_int height,
				u_int mul,   u_int div)		;
	    DC&		setZoom(u_int mul,   u_int div)		;
    virtual DC&		setOffset(int u0, int v0)		;
    virtual DC&		setLayer(Layer layer)			;
	    DC&		setPointStyle(PointStyle pointStyle)	;
    virtual DC&		setThickness(u_int thickness)		= 0;
    virtual DC&		setForeground(const BGR& fg)		= 0;
    virtual DC&		setBackground(const BGR& bg)		= 0;
    virtual DC&		setForeground(u_int index)		= 0;
    virtual DC&		setBackground(u_int index)		= 0;
    virtual DC&		setSaturation(u_int saturation)		= 0;
    
    virtual DC&		clear()					= 0;
	    DC&		repaint()				;
	    DC&		repaintAll()				;
    virtual DC&		sync()					= 0;
    
    virtual DC&		operator <<(const Point2<int>& p)	= 0;
    template <class T>
	    DC&		operator <<(const Point2<T>& p)		;
    virtual DC&		operator <<(const LineP2d& p)		= 0;
    virtual DC&		operator <<(const Image<u_char>& image)	= 0;
    virtual DC&		operator <<(const Image<s_char>& image)	= 0;
    virtual DC&		operator <<(const Image<short>&  image)	= 0;
    virtual DC&		operator <<(const Image<BGR>&  image)	= 0;
    virtual DC&		operator <<(const Image<ABGR>& image)	= 0;
    virtual DC&		operator <<(const Image<YUV444>& image)	= 0;
    virtual DC&		operator <<(const Image<YUV422>& image)	= 0;
    virtual DC&		operator <<(const Image<YUV411>& image)	= 0;
    virtual DC&		draw(const char* s, int u, int v)	= 0;

    int			log2devR(int r)			const	;
    int			log2devU(int u)			const	;
    int			log2devV(int v)			const	;
    Point2<int>		log2dev(const Point2<int>& p)	const	;
    int			dev2logR(int r)			const	;
    int			dev2logU(int u)			const	;
    int			dev2logV(int v)			const	;
    Point2<int>		dev2log(const Point2<int>& p)	const	;

  protected:
    enum		{PRADIUS = 5};		// radius of marker points
	    
    int			hasScale()		const	{return _mul != _div;}
    u_int		deviceWidth()		const	;
    u_int		deviceHeight()		const	;

    Layer		getLayer()		const	;
    PointStyle		getPointStyle()		const	;
    virtual u_int	getThickness()		const	= 0;
    
    virtual DC&		repaintUnderlay()		= 0;
    virtual DC&		repaintOverlay()		= 0;
    
  private:
    u_int		_width;		// logical width  of DC
    u_int		_height;	// logical height of DC
    u_int		_mul, _div;	// zooming factors 
    Point2<int>		_offset;	// logical coordinates of the
					// temporary offset
    Layer		_layer;		// flag indicating underlay/overlay
    PointStyle		_pointStyle;	// drawing style for points
};

inline DC&
DC::setZoom(u_int mul, u_int div)
{
    return setSize(width(), height(), mul, div);
}
    
template <class T> inline DC&
DC::operator <<(const Point2<T>& p)
{
    return *this << Point2<int>(p);
}

inline int
DC::log2devR(int r)	const	{return r * _mul / _div;}

inline int
DC::log2devU(int u)	const	{return
				   log2devR(u + _offset[0]) + _mul/(2*_div);}

inline int
DC::log2devV(int v)	const	{return
				   log2devR(v + _offset[1]) + _mul/(2*_div);}

inline Point2<int>
DC::log2dev(const Point2<int>& p) const
{
    if (hasScale())
	return Point2<int>(log2devU(p[0]), log2devV(p[1]));
    else
	return Point2<int>(p[0] + _offset[0], p[1] + _offset[1]);
}

inline int
DC::dev2logR(int r)	const	{return r * _div / _mul;}

inline int
DC::dev2logU(int u)	const	{return
				   dev2logR(u - _mul/(2*_div)) - _offset[0];}

inline int
DC::dev2logV(int v)	const	{return
				   dev2logR(v - _mul/(2*_div)) - _offset[1];}

inline Point2<int>
DC::dev2log(const Point2<int>& p) const
{
    if (hasScale())
	return Point2<int>(dev2logU(p[0]), dev2logV(p[1]));
    else
	return Point2<int>(p[0] - _offset[0], p[1] - _offset[1]);
}

inline u_int
DC::deviceWidth()	const	{return log2devR(_width);}

inline u_int
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
extern DC&			x0_25(DC&)		;
extern DC&			x0_5(DC&)		;
extern DC&			x1(DC&)			;
extern DC&			x1_5(DC&)		;
extern DC&			x2(DC&)			;
extern DC&			x4(DC&)			;
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
extern OManip1<DC, u_int>	foreground(u_int)	;
extern OManip1<DC, u_int>	background(u_int)	;
extern OManip1<DC, u_int>	thickness(u_int)	;
extern OManip1<DC, u_int>	saturation(u_int)	;
extern OManip2<DC, int, int>	offset(int, int)	;

template <class S> inline S&
operator <<(S& dc, DC& (*f)(DC&))
{
    (*f)(dc);
    return dc;
}

}
}
#endif	// !__TUvDC_h
