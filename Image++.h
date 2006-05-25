/*
 *  平9年 電子技術総合研究所 植芝俊夫 著作権所有
 *
 *  著作者による許可なしにこのプログラムの第三者への開示、複製、改変、
 *  使用等その他の著作人格権を侵害する行為を禁止します。
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *
 *  Copyright 1996
 *  Toshio UESHIBA, Electrotechnical Laboratory
 *
 *  All rights reserved.
 *  Any changing, copying or giving information about source programs of
 *  any part of this software and/or documentation without permission of the
 *  authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damage in use of this program.
 */

/*
 *  $Id: Image++.h,v 1.13 2006-05-25 02:10:17 ueshiba Exp $
 */
#ifndef	__TUImagePP_h
#define	__TUImagePP_h

#include <string.h>
#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class RGB, BGR, RGBA & ABGR						*
*	Note:	X::operator =(const X&) must be explicitly defined	*
*		to avoid X -> double -> u_char -> X conversion.		*
************************************************************************/
struct BGR;
struct YUV444;
struct RGB
{
    RGB()					:r(0), g(0), b(0)	{}
    RGB(u_char c)				:r(c), g(c), b(c)	{}
    RGB(u_char rr, u_char gg, u_char bb)	:r(rr), g(gg), b(bb)	{}
    RGB(const BGR&)				;
    RGB(const YUV444&)				;

		operator u_char()	const	{return u_char(double(*this));}
		operator short()	const	{return short(double(*this));}
		operator float()	const	{return float(double(*this));}
		operator double()	const	{return 0.3*r+0.59*g+0.11*b+0.5;}
    
    RGB&	operator +=(const RGB& v)	{r += v.r; g += v.g; b += v.b;
						 return *this;}
    RGB&	operator -=(const RGB& v)	{r -= v.r; g -= v.g; b -= v.b;
						 return *this;}
    bool	operator ==(const RGB& v) const	{return (r == v.r &&
							 g == v.g &&
							 b == v.b);}
    bool	operator !=(const RGB& v) const	{return !(*this == v);}
    
    u_char	r, g, b;
};

inline std::istream&
operator >>(std::istream& in, RGB& v)
{
    return in >> (u_int&)v.r >> (u_int&)v.g >> (u_int&)v.b;
}

inline std::ostream&
operator <<(std::ostream& out, const RGB& v)
{
    return out << (u_int)v.r << ' ' << (u_int)v.g << ' ' << (u_int)v.b;
}

struct BGR
{
    BGR()					:b(0),   g(0),   r(0)	{}
    BGR(u_char c)				:b(c),   g(c),   r(c)	{}
    BGR(u_char rr, u_char gg, u_char bb)	:b(bb),  g(gg),  r(rr)	{}
    BGR(const RGB& v)				:b(v.b), g(v.g), r(v.r)	{}
    BGR(const YUV444&)				;

		operator u_char()	const	{return u_char(double(*this));}
		operator short()	const	{return short(double(*this));}
		operator float()	const	{return float(double(*this));}
		operator double()	const	{return 0.3*r+0.59*g+0.11*b+0.5;}

    BGR&	operator +=(const BGR& v)	{r += v.r; g += v.g; b += v.b;
						 return *this;}
    BGR&	operator -=(const BGR& v)	{r -= v.r; g -= v.g; b -= v.b;
						 return *this;}
    bool	operator ==(const BGR& v) const	{return (r == v.r &&
							 g == v.g &&
							 b == v.b);}
    bool	operator !=(const BGR& v) const	{return !(*this != v);}
    
    u_char	b, g, r;
};

inline
RGB::RGB(const BGR& v)	:r(v.r), g(v.g), b(v.b)	{}

inline std::istream&
operator >>(std::istream& in, BGR& v)
{
    return in >> (u_int&)v.r >> (u_int&)v.g >> (u_int&)v.b;
}

inline std::ostream&
operator <<(std::ostream& out, const BGR& v)
{
    return out << (u_int)v.r << ' ' << (u_int)v.g << ' ' << (u_int)v.b;
}

struct Alpha
{
    Alpha()	:a(0)					{}

    int		operator ==(const Alpha& v) const	{return a == v.a;}
    int		operator !=(const Alpha& v) const	{return !(*this == v);}
    
    u_char	a;
};

struct RGBA : public RGB, public Alpha
{
    RGBA()			:RGB(),  Alpha()  	{}
    RGBA(u_char c)		:RGB(c), Alpha()  	{}
    RGBA(const RGB& v)		:RGB(v), Alpha()  	{}
    RGBA(const BGR& v)		:RGB(v), Alpha()  	{}
    RGBA(const RGBA& v)		:RGB(v), Alpha(v) 	{}
    RGBA(const YUV444& v)				;

    bool	operator ==(const RGBA& v) const
		{return (Alpha::operator ==(v) && RGB::operator ==(v));}
    bool	operator !=(const RGBA& v)	const	{return !(*this != v);}
};

struct ABGR : public Alpha, public BGR
{
    ABGR()			:Alpha(),  BGR()	{}
    ABGR(u_char c)		:Alpha(),  BGR(c)	{}
    ABGR(const BGR& v)		:Alpha(),  BGR(v)	{}
    ABGR(const RGB& v)		:Alpha(),  BGR(v)	{}
    ABGR(const ABGR& v)		:Alpha(v), BGR(v)	{}
    ABGR(const YUV444& v)				;

    bool	operator ==(const ABGR& v) const
		{return (Alpha::operator ==(v) && BGR::operator ==(v));}
    bool	operator !=(const ABGR& v) const	{return !(*this != v);}
};

/************************************************************************
*  class YUV444, YUV422, YUV411						*
************************************************************************/
struct YUV444
{
    YUV444(u_char yy=0, u_char uu=128, u_char vv=128)
	:u(uu),  y(yy), v(vv)				{}
    YUV444(const RGB& v)	:u(128), y(v), v(128)	{}
    YUV444(const BGR& v)	:u(128), y(v), v(128)	{}

		operator u_char()		 const	{return u_char(y);}
		operator short()		 const	{return short(y);}
		operator float()		 const	{return float(y);}
		operator double()		 const	{return double(y);}
    bool	operator ==(const YUV444& yuv) const	{return (u == yuv.u &&
								 y == yuv.y &&
								 v == yuv.v);}
    bool	operator !=(const YUV444& yuv) const	{return !(*this==yuv);}
    

    u_char	u, y, v;
};
    
inline std::istream&
operator >>(std::istream& in, YUV444& yuv)
{
    return in >> (u_int&)yuv.y >> (u_int&)yuv.u >> (u_int&)yuv.v;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV444& yuv)
{
    return out << (u_int)yuv.y << ' ' << (u_int)yuv.u << ' ' << (u_int)yuv.v;
}

struct YUV422
{
    YUV422(u_char yy=0, u_char xx=128) :x(xx),  y(yy)	{}
    YUV422(const RGB& v)		 :x(128), y(v)	{}
    YUV422(const BGR& v)		 :x(128), y(v)	{}

		operator u_char()		const	{return u_char(y);}
		operator short()		const	{return short(y);}
		operator float()		const	{return float(y);}
		operator double()		const	{return double(y);}
    bool	operator ==(const YUV422& v)	const	{return (x == v.x &&
								 y == v.y);}
    bool	operator !=(const YUV422& v)	const	{return !(*this == v);}
    
    u_char	x, y;
};

inline std::istream&
operator >>(std::istream& in, YUV422& yuv)
{
    return in >> (u_int&)yuv.y >> (u_int&)yuv.x;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV422& yuv)
{
    return out << (u_int)yuv.y << ' ' << (u_int)yuv.x;
}

struct YUV411
{
    YUV411(u_char yy0=0, u_char yy1=0, u_char xx=128)
			      :x(xx),  y0(yy0), y1(yy1)	{}
    YUV411(const RGB& v)  :x(128), y0(v),   y1(v)	{}
    YUV411(const BGR& v)  :x(128), y0(v),   y1(v)	{}

    bool	operator ==(const YUV411& v)	const	{return (x  == v.x  &&
								 y0 == v.y0 &&
								 y1 == v.y1);}
    bool	operator !=(const YUV411& v)	const	{return !(*this == v);}
    
    u_char	x, y0, y1;
};

inline std::istream&
operator >>(std::istream& in, YUV411& yuv)
{
    return in >> (u_int&)yuv.y0 >> (u_int&)yuv.y1 >> (u_int&)yuv.x;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV411& yuv)
{
    return out << (u_int)yuv.y0 << ' ' << (u_int)yuv.y1 << ' ' << (u_int)yuv.x;
}

/************************************************************************
*  function fromYUV<T>()						*
************************************************************************/
class ConversionFromYUV
{
  public:
    ConversionFromYUV()					;

  private:
    template <class T>
    friend T	fromYUV(u_char y, u_char u, u_char v)	;
    
    int		_r[256], _g0[256], _g1[256], _b[256];
};

extern const ConversionFromYUV	conversionFromYUV;

template <class T> inline T
fromYUV(u_char y, u_char u, u_char v)
{
    T	val;
    int	tmp = y + conversionFromYUV._r[v];
    val.r = (tmp > 255 ? 255 : tmp < 0 ? 0 : tmp);
    tmp   =
	y - (int(conversionFromYUV._g0[v] + conversionFromYUV._g1[u]) >> 10);
    val.g = (tmp > 255 ? 255 : tmp < 0 ? 0 : tmp);
    tmp   = y + conversionFromYUV._b[u];
    val.b = (tmp > 255 ? 255 : tmp < 0 ? 0 : tmp);
    return val;
}

template <> inline u_char
fromYUV<u_char>(u_char y, u_char, u_char)
{
    return y;
}

template <> inline short
fromYUV<short>(u_char y, u_char, u_char)
{
    return y;
}

template <> inline float
fromYUV<float>(u_char y, u_char, u_char)
{
    return y;
}

template <> inline double
fromYUV<double>(u_char y, u_char, u_char)
{
    return y;
}

template <> inline YUV444
fromYUV<YUV444>(u_char y, u_char u, u_char v)
{
    return YUV444(y, u, v);
}

inline
RGB::RGB(const YUV444& v)
{
    *this = fromYUV<RGB>(v.y, v.u, v.v);
}

inline
BGR::BGR(const YUV444& v)
{
    *this = fromYUV<BGR>(v.y, v.u, v.v);
}

inline
RGBA::RGBA(const YUV444& v)
     :RGB(v), Alpha()
{
}

inline
ABGR::ABGR(const YUV444& v)
     :Alpha(),  BGR(v)
{
}

/************************************************************************
*  class ImageBase:	basic image class				*
************************************************************************/
class ImageBase
{
  protected:
    ImageBase()
	:P(3, 4), d1(0), d2(0)		{P[0][0] = P[1][1] = P[2][2] = 1.0;}
    virtual ~ImageBase()		;
    
  public:
    enum Type		{END = 0, U_CHAR = 5, RGB_24 = 6,
			 SHORT, FLOAT, DOUBLE,
			 YUV_444, YUV_422, YUV_411};
    
    Type		restoreHeader(std::istream& in)			;
    std::ostream&	saveHeader(std::ostream& out, Type type) const	;

    u_int		width()			const	{return _width();}
    u_int		height()		const	{return _height();}
    void		resize(u_int h, u_int w)	{_resize(h, w);}
	
  private:
    virtual u_int	_width()		const	= 0;
    virtual u_int	_height()		const	= 0;
    virtual void	_resize(u_int h, u_int w)	= 0;

    static u_int	type2depth(Type type)		;
    
  public:
    Matrix<double>	P;			// projection matrix
    double		d1, d2;			// distortion parameters
};

/************************************************************************
*  class ImageLine<T>:	Generic image scanline class			*
************************************************************************/
template <class T>
class ImageLine : public Array<T>
{
  public:
    explicit ImageLine(u_int d=0)
        :Array<T>(d), _lmost(0), _rmost(d)		{*this = 0;}
    ImageLine(T* p, u_int d)
        :Array<T>(p, d), _lmost(0), _rmost(d)		{}
    ImageLine&		operator =(double c)
			{
			    Array<T>::operator =(c);
			    return *this;
			}

    using		Array<T>::dim;
    const YUV422*	fill(const YUV422* src)		;
    const YUV411*	fill(const YUV411* src)		;
    const T*		fill(const T* src)		;
    template <class S>
    const S*		fill(const S* src)		;
    int			lmost()			const	{return _lmost;}
    int			rmost()			const	{return _rmost;}
    void		setLimits(int l, int r)		{_lmost = l;
							 _rmost = r;}
    bool		valid(int u)		const	{return (u >= _lmost &&
								 u <  _rmost);}
	
    bool		resize(u_int d)			;
    void		resize(T* p, u_int d)		;

  private:
    int			_lmost;
    int			_rmost;
};

template <class T> inline const T*
ImageLine<T>::fill(const T* src)
{
    memcpy((T*)*this, src, dim() * sizeof(T));
    return src + dim();
}

template <class T> inline bool
ImageLine<T>::resize(u_int d)
{
    _lmost = 0;
    _rmost = d;
    return Array<T>::resize(d);
}

template <class T> inline void
ImageLine<T>::resize(T* p, u_int d)
{
    _lmost = 0;
    _rmost = d;
    Array<T>::resize(p, d);
}

template <>
class ImageLine<YUV422> : public Array<YUV422>
{
  public:
    explicit ImageLine(u_int d=0)
	:Array<YUV422>(d), _lmost(0), _rmost(d)		{*this = 0;}
    ImageLine(YUV422* p, u_int d)
	:Array<YUV422>(p, d), _lmost(0), _rmost(d)	{}
    ImageLine&		operator =(double c)
			{
			    Array<YUV422>::operator =(c);
			    return *this;
			}
    const YUV444*	fill(const YUV444* src)		;
    const YUV422*	fill(const YUV422* src)		;
    const YUV411*	fill(const YUV411* src)		;
    template <class S>
    const S*		fill(const S* src)		;
    int			lmost()			const	{return _lmost;}
    int			rmost()			const	{return _rmost;}
    void		setLimits(int l, int r)		{_lmost = l;
							 _rmost = r;}
    bool		valid(int u)		const	{return (u >= _lmost &&
								 u <  _rmost);}
	
    bool		resize(u_int d)			;
    void		resize(YUV422* p, u_int d)	;

  private:
    int			_lmost;
    int			_rmost;
};

inline const YUV422*
ImageLine<YUV422>::fill(const YUV422* src)
{
    memcpy((YUV422*)*this, src, dim() * sizeof(YUV422));
    return src + dim();
}

inline bool
ImageLine<YUV422>::resize(u_int d)
{
    _lmost = 0;
    _rmost = d;
    return Array<YUV422>::resize(d);
}

inline void
ImageLine<YUV422>::resize(YUV422* p, u_int d)
{
    _lmost = 0;
    _rmost = d;
    Array<YUV422>::resize(p, d);
}

template <>
class ImageLine<YUV411> : public Array<YUV411>
{
  public:
    explicit ImageLine(u_int d=0)
	:Array<YUV411>(d), _lmost(0), _rmost(d)		{*this = 0;}
    ImageLine(YUV411* p, u_int d)
	:Array<YUV411>(p, d), _lmost(0), _rmost(d)	{}
    ImageLine&		operator =(double c)
			{
			    Array<YUV411>::operator =(c);
			    return *this;
			}
    const YUV444*	fill(const YUV444* src)		;
    const YUV422*	fill(const YUV422* src)		;
    const YUV411*	fill(const YUV411* src)		;
    template <class S>
    const S*		fill(const S* src)		;
    int			lmost()			const	{return _lmost;}
    int			rmost()			const	{return _rmost;}
    void		setLimits(int l, int r)		{_lmost = l;
							 _rmost = r;}
    bool		valid(int u)		const	{return (u >= _lmost &&
								 u <  _rmost);}
	
    bool		resize(u_int d)			;
    void		resize(YUV411* p, u_int d)	;

  private:
    int			_lmost;
    int			_rmost;
};

inline const YUV411*
ImageLine<YUV411>::fill(const YUV411* src)
{
    memcpy((YUV411*)*this, src, dim() * sizeof(YUV411));
    return src + dim();
}

inline bool
ImageLine<YUV411>::resize(u_int d)
{
    _lmost = 0;
    _rmost = d;
    return Array<YUV411>::resize(d);
}

inline void
ImageLine<YUV411>::resize(YUV411* p, u_int d)
{
    _lmost = 0;
    _rmost = d;
    Array<YUV411>::resize(p, d);
}

/************************************************************************
*  class Image<T>:	Generic image class				*
************************************************************************/
template <class T>
class Image : public Array2<ImageLine<T> >, public ImageBase
{
  public:
    explicit Image(u_int w=0, u_int h=0)
	:Array2<ImageLine<T> >(h, w), ImageBase()		{*this = 0;}
    Image(T* p, u_int w, u_int h)			
	:Array2<ImageLine<T> >(p, h, w), ImageBase()		{}
    Image(const Image& i, u_int u, u_int v, u_int w, u_int h)
	:Array2<ImageLine<T> >(i, v, u, h, w), ImageBase(i)	{}
    Image(const Image& i)			
	:Array2<ImageLine<T> >(i), ImageBase(i)			{}
    Image&	operator =(const Image& i)	{Array2<ImageLine<T> >::
						 operator =(i);
						 (ImageBase&)*this = i;
						 return *this;}

    u_int	width()			const	{return
						 Array2<ImageLine<T> >::ncol();}
    u_int	height()		const	{return
						 Array2<ImageLine<T> >::nrow();}
    
    Image&	operator = (double c)		{Array2<ImageLine<T> >::
						 operator  =(c); return *this;}
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out, Type type)	const	;
    std::istream&	restoreData(std::istream& in, Type type)	;
    std::ostream&	saveData(std::ostream& out, Type type)	const	;
    void		resize(u_int h, u_int w)			;
    void		resize(T* p, u_int h, u_int w)			;

  private:
    template <class S>
    std::istream&	restoreRows(std::istream& in)			;
    template <class D>
    std::ostream&	saveRows(std::ostream& out)		const	;

    virtual u_int	_width()				const	;
    virtual u_int	_height()				const	;
    virtual void	_resize(u_int h, u_int w)			;
};

template <class T> inline std::istream&
Image<T>::restore(std::istream& in)
{
    return restoreData(in, restoreHeader(in));
}

template <class T> inline std::ostream&
Image<T>::save(std::ostream& out, Type type) const
{
    saveHeader(out, type);
    return saveData(out, type);
}

template <class T> inline void
Image<T>::resize(u_int h, u_int w)
{
    Array2<ImageLine<T> >::resize(h, w);
}

template <class T> inline void
Image<T>::resize(T* p, u_int h, u_int w)
{
    Array2<ImageLine<T> >::resize(p, h, w);
}
 
template <> inline
Image<YUV411>::Image(u_int w, u_int h)
    :Array2<ImageLine<TU::YUV411> >(h, w/2), ImageBase()
{
    *this = 0.0;
}

template <> inline
Image<YUV411>::Image(TU::YUV411* p, u_int w, u_int h)
    :Array2<ImageLine<TU::YUV411> >(p, h, w/2), ImageBase()
{
}

template <> inline
Image<YUV411>::Image(const Image& i, u_int u, u_int v, u_int w, u_int h)
    :Array2<ImageLine<TU::YUV411> >(i, v, u/2, h, w/2), ImageBase(i)
{
}

template <> inline u_int
Image<YUV411>::width() const
{
    return 2 * ncol();
}

template <> inline void
Image<YUV411>::resize(TU::YUV411* p, u_int h, u_int w)
{
    Array2<ImageLine<YUV411> >::resize(p, h, w/2);
}

}

#endif	/* !__TUImagePP_h */
