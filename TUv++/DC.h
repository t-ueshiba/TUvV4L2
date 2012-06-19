/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
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
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id: DC.h,v 1.16 2012-06-19 08:33:48 ueshiba Exp $  
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
    DC(u_int width, u_int height, u_int mul, u_int div)
	:_width(width), _height(height),
	 _mul(mul), _div(div), _offset(0, 0),
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
    virtual DC&		setSaturationF(float saturation)	= 0;
    
    virtual DC&		clear()					= 0;
	    DC&		repaint()				;
	    DC&		repaintAll()				;
    virtual DC&		sync()					= 0;
    
    virtual DC&		operator <<(const Point2<int>& p)	= 0;
    template <class T>
	    DC&		operator <<(const Point2<T>& p)		;
    virtual DC&		operator <<(const LineP2f& l)		= 0;
    virtual DC&		operator <<(const LineP2d& l)		= 0;
    virtual DC&		operator <<(const Image<u_char>& image)	= 0;
    virtual DC&		operator <<(const Image<s_char>& image)	= 0;
    virtual DC&		operator <<(const Image<short>&  image)	= 0;
    virtual DC&		operator <<(const Image<float>&  image)	= 0;
    virtual DC&		operator <<(const Image<BGR>&    image)	= 0;
    virtual DC&		operator <<(const Image<ABGR>&   image)	= 0;
    virtual DC&		operator <<(const Image<BGRA>&   image)	= 0;
    virtual DC&		operator <<(const Image<RGB>&    image)	= 0;
    virtual DC&		operator <<(const Image<RGBA>&   image)	= 0;
    virtual DC&		operator <<(const Image<ARGB>&   image)	= 0;
    virtual DC&		operator <<(const Image<YUV444>& image)	= 0;
    virtual DC&		operator <<(const Image<YUV422>& image)	= 0;
    virtual DC&		operator <<(const Image<YUYV422>& image)= 0;
    virtual DC&		operator <<(const Image<YUV411>& image)	= 0;
    virtual DC&		drawLine(const Point2<int>& p,
				 const Point2<int>& q)		= 0;
    template <class T>
	    DC&		drawLine(const Point2<T>& p,
				 const Point2<T>& q)		;
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

template <class T> inline DC&
DC::drawLine(const Point2<T>& p, const Point2<T>& q)
{
    return drawLine(Point2<int>(p), Point2<int>(q));
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
#endif	// !__TUvDC_h
