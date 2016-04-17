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
 *  $Id$
 */
/*!
  \file		Image++.h
  \brief	画素と画像に関連するクラスの定義と実装
*/
#ifndef	__TU_IMAGEPP_H
#define	__TU_IMAGEPP_H

#include <string.h>
#include <boost/operators.hpp>
#include "TU/types.h"
#include "TU/Geometry++.h"

namespace TU
{
namespace detail
{
/************************************************************************
*  class detail::ColorConverter						*
************************************************************************/
class ColorConverter
{
  private:
    constexpr static float	_yr = 0.299f;		// ITU-R BT.601, PAL
    constexpr static float	_yb = 0.114f;		// ITU-R BT.601, PAL
    constexpr static float	_yg = 1.0f - _yr - _yb;	// ITU-R BT.601, PAL
    constexpr static float	_ku = 0.4921f;		// ITU-R BT.601, PAL
    constexpr static float	_kv = 0.877314f;	// ITU-R BT.601, PAL
    
  public:
		ColorConverter()					;

    int		r(int y, int v) const
		{
		    return limit(y + _r[v]);
		}
    int		g(int y, int u, int v) const
		{
		    return limit(y - scaleDown(_gu[u] - _gv[v]));
		}
    int		b(int y, int u) const
		{
		    return limit(y + _b[u]);
		}
    template <class T>
    static T	y(int r, int g, int b)
		{
		    return T(_yr*r + _yg*g + _yb*b);
		}
    int		u(int b, int y) const
		{
		    return _u[255 + b - y];
		}
    int		v(int r, int y) const
		{
		    return _v[255 + r - y];
		}
    
  private:
    template <class T>
    static int	limit(T val)
		{
		    return (val < 0 ? 0 : val > 255 ? 255 : int(val));
		}
    static int	scaleUp(float val)
		{
		    return int(val * (1 << 10));
		}
    static int	scaleDown(int val)
		{
		    return val >> 10;
		}
    
  private:
    int		_u[255 + 1 + 255];
    int		_v[255 + 1 + 255];

    int		_r[256];
    int		_gu[256];
    int		_gv[256];
    int		_b[256];
};

extern const ColorConverter	colorConverter;
    
/************************************************************************
*  struct detail::[RGB|BGR|RGBA|ARGB|ABGR|BGRA]				*
************************************************************************/
struct RGB
{
    typedef u_char	element_type;
    
    RGB(element_type rr, element_type gg, element_type bb)
	:r(rr), g(gg), b(bb)						{}
    
    element_type r, g, b;
};

struct BGR
{
    typedef u_char	element_type;
    
    BGR(element_type rr, element_type gg, element_type bb)
	:b(bb), g(gg), r(rr)						{}
    
    element_type b, g, r;
};

struct RGBA
{
    typedef u_char	element_type;
    
    RGBA(element_type rr, element_type gg, element_type bb,
	 element_type aa=255)	:r(rr), g(gg), b(bb), a(aa)		{}
    
    element_type r, g, b, a;
};

struct ABGR
{
    typedef u_char	element_type;
    
    ABGR(element_type rr, element_type gg, element_type bb,
	 element_type aa=255)	:a(aa), b(bb), g(gg), r(rr)		{}
    
    element_type a, b, g, r;
};

struct ARGB
{
    typedef u_char	element_type;
    
    ARGB(element_type rr, element_type gg, element_type bb,
	 element_type aa=255)	:a(aa), r(rr), g(gg), b(bb)		{}
    
    element_type a, r, g, b;
};

struct BGRA
{
    typedef u_char	element_type;
    
    BGRA(element_type rr, element_type gg, element_type bb,
	 element_type aa=255)	:b(bb), g(gg), r(rr), a(aa)		{}
    
    element_type b, g, r, a;
};
}	// namespace detail

/************************************************************************
*  struct RGB_<E>							*
************************************************************************/
struct YUV444;
    
template <class E>
struct RGB_ : public E, boost::additive<RGB_<E>,
			boost::multiplicative<RGB_<E>, float,
			boost::equality_comparable<RGB_<E> > > >
{
    typedef typename E::element_type	element_type;
    
    RGB_()				:E(0, 0, 0)			{}
    RGB_(element_type rr, element_type gg, element_type bb)
					:E(rr, gg, bb)			{}
    RGB_(element_type rr, element_type gg, element_type bb,
	 element_type aa)		:E(rr, gg, bb, aa)		{}
    RGB_(const RGB_<detail::RGB>& p)	:E(p.r, p.g, p.b)		{}
    RGB_(const RGB_<detail::BGR>& p)	:E(p.r, p.g, p.b)		{}
    template <class E_>
    RGB_(const RGB_<E_>& p)		:E(p.r, p.g, p.b, p.a)		{}
    template <class T_>
    RGB_(const T_& p)
	:E(element_type(p), element_type(p), element_type(p))		{}
    RGB_(const YUV444& p)						;
    
    using	E::r;
    using	E::g;
    using	E::b;

    template <class T_,
	      typename std::enable_if<std::is_arithmetic<T_>::value>::type*
	      = nullptr>
		operator T_() const
		{
		    return detail::colorConverter.y<T_>(r, g, b);
		}
    RGB_&	operator +=(const RGB_& p)
		{
		    r += p.r; g += p.g; b += p.b;
		    return *this;
		}
    RGB_&	operator -=(const RGB_& p)
		{
		    r -= p.r; g -= p.g; b -= p.b;
		    return *this;
		}
    RGB_&	operator *=(float c)
		{
		    r *= c; g *= c; b *= c;
		    return *this;
		}
    RGB_&	operator /=(float c)
		{
		    r /= c; g /= c; b /= c;
		    return *this;
		}
    bool	operator ==(const RGB_& p) const
		{
		    return (r == p.r && g == p.g && b == p.b && E::a == p.a);
		}
};

template <class E> inline std::istream&
operator >>(std::istream& in, RGB_<E>& p)
{
    return in >> (u_int&)p.r >> (u_int&)p.g >> (u_int&)p.b >> (u_int&)p.a;
}

template <class E> inline std::ostream&
operator <<(std::ostream& out, const RGB_<E>& p)
{
    return out << u_int(p.r) << ' ' << u_int(p.g) << ' ' << u_int(p.b) << ' '
	       << u_int(p.a);
}
    
/************************************************************************
*  struct RGB								*
************************************************************************/
//! Red, Green, Blue（各8bit）の順で並んだカラー画素
typedef RGB_<detail::RGB>	RGB;

template <> template <class E1> inline
RGB_<detail::RGB>::RGB_(const RGB_<E1>& p) :detail::RGB(p.r, p.g, p.b)	{}
    
template <> inline bool
RGB_<detail::RGB>::operator ==(const RGB_& p) const
{
    return (r == p.r && g == p.g && b == p.b);
}

inline std::istream&
operator >>(std::istream& in, RGB& p)
{
    return in >> (u_int&)p.r >> (u_int&)p.g >> (u_int&)p.b;
}

inline std::ostream&
operator <<(std::ostream& out, const RGB& p)
{
    return out << u_int(p.r) << ' ' << u_int(p.g) << ' ' << u_int(p.b);
}

/************************************************************************
*  struct BGR								*
************************************************************************/
//! Blue, Green, Red（各8bit）の順で並んだカラー画素
typedef RGB_<detail::BGR>	BGR;

template <> template <class E1> inline
RGB_<detail::BGR>::RGB_(const RGB_<E1>& p) :detail::BGR(p.r, p.g, p.b)	{}
    
template <> inline bool
RGB_<detail::BGR>::operator ==(const RGB_& p) const
{
    return (r == p.r && g == p.g && b == p.b);
}

inline std::istream&
operator >>(std::istream& in, BGR& p)
{
    return in >> (u_int&)p.r >> (u_int&)p.g >> (u_int&)p.b;
}

inline std::ostream&
operator <<(std::ostream& out, const BGR& p)
{
    return out << u_int(p.r) << ' ' << u_int(p.g) << ' ' << u_int(p.b);
}

/************************************************************************
*  struct RGBA, ABGR, ARGB, BGRA					*
************************************************************************/
//! Red, Green, Blue, Alpha（各8bit）の順で並んだカラー画素
typedef RGB_<detail::RGBA>	RGBA;
//! Alpha, Blue, Green, Red（各8bit）の順で並んだカラー画素
typedef RGB_<detail::ABGR>	ABGR;
//! Alpha, Red, Green, Blue（各8bit）の順で並んだカラー画素
typedef RGB_<detail::ARGB>	ARGB;
//! Blue, Green, Red, Alpha（各8bit）の順で並んだカラー画素
typedef RGB_<detail::BGRA>	BGRA;

/************************************************************************
*  struct YUV444, YUV422, YUYV422, YUV411				*
************************************************************************/
//! U, Y, V（各8bit）の順で並んだカラー画素
struct YUV444
{
    typedef u_char	element_type;
    
    YUV444(element_type yy=0, element_type uu=128, element_type vv=128)
	:u(uu), y(yy), v(vv)						{}
    template <class E>
    YUV444(const RGB_<E>& p)
	:y(detail::colorConverter.y<element_type>(p.r, p.g, p.b))
		{
		    u = detail::colorConverter.u(p.b, y);
		    v = detail::colorConverter.v(p.r, y);
		}
    template <class T>
    YUV444(const T& p)	:u(128), y(p), v(128)				{}
    
    template <class T,
	      typename std::enable_if<std::is_arithmetic<T>::value>::type*
	      = nullptr>
		operator T()			const	{return T(y);}
    bool	operator ==(const YUV444& yuv)	const	{return (u == yuv.u &&
								 y == yuv.y &&
								 v == yuv.v);}
    bool	operator !=(const YUV444& yuv)	const	{return !(*this==yuv);}

    element_type	u, y, v;
};
    
inline std::istream&
operator >>(std::istream& in, YUV444& yuv)
{
    return in >> (u_int&)yuv.y >> (u_int&)yuv.u >> (u_int&)yuv.v;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV444& yuv)
{
    return out << u_int(yuv.y) << ' ' << u_int(yuv.u) << ' ' << u_int(yuv.v);
}

template <class E> inline
RGB_<E>::RGB_(const YUV444& p)
    :E(detail::colorConverter.r(p.y, p.v),
       detail::colorConverter.g(p.y, p.u, p.v),
       detail::colorConverter.b(p.y, p.u))
{
}

struct YUYV422;

//! [U, Y0], [V, Y1]（各8bit）の順で並んだカラー画素(16bits/pixel)
struct YUV422
{
    typedef u_char	element_type;

    YUV422(element_type yy=0, element_type xx=128)	:x(xx), y(yy)	{}
    YUV422(const YUYV422& p)						;

    template <class T,
	      typename std::enable_if<std::is_arithmetic<T>::value>::type*
	      = nullptr>
		operator T()			const	{return T(y);}
    bool	operator ==(const YUV422& p)	const	{return (x == p.x &&
								 y == p.y);}
    bool	operator !=(const YUV422& p)	const	{return !(*this == p);}
    
    element_type	x, y;
};

inline std::istream&
operator >>(std::istream& in, YUV422& yuv)
{
    return in >> (u_int&)yuv.y >> (u_int&)yuv.x;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV422& yuv)
{
    return out << u_int(yuv.y) << ' ' << u_int(yuv.x);
}

//! [Y0, U], [Y1, V]（各8bit）の順で並んだカラー画素(16bits/pixel)
struct YUYV422
{
    typedef u_char	element_type;

    YUYV422(element_type yy=0, element_type xx=128)	:y(yy), x(xx)	{}
    YUYV422(const YUV422& p)				:y(p.y), x(p.x)	{}

    template <class T,
	      typename std::enable_if<std::is_arithmetic<T>::value>::type*
	      = nullptr>
		operator T()			const	{return T(y);}
    bool	operator ==(const YUYV422& p)	const	{return (y == p.y &&
								 x == p.x);}
    bool	operator !=(const YUYV422& p)	const	{return !(*this == p);}
    
    element_type	y, x;
};

inline std::istream&
operator >>(std::istream& in, YUYV422& yuv)
{
    return in >> (u_int&)yuv.y >> (u_int&)yuv.x;
}

inline std::ostream&
operator <<(std::ostream& out, const YUYV422& yuv)
{
    return out << u_int(yuv.y) << ' ' << u_int(yuv.x);
}

inline
YUV422::YUV422(const YUYV422& p)  :x(p.x), y(p.y)	{}

//! [U, Y0, Y1], [V, Y2, Y3]（各8bit）の順で並んだカラー画素(12bits/pixel)
struct YUV411
{
    typedef u_char	element_type;

    YUV411(element_type yy0=0, element_type yy1=0, element_type xx=128)
	:x(xx), y0(yy0), y1(yy1)					{}
    template <class T>
    YUV411(const T& p)	:x(128), y0(p), y1((&p)[1])			{}

    bool	operator ==(const YUV411& p)	const	{return (x  == p.x  &&
								 y0 == p.y0 &&
								 y1 == p.y1);}
    bool	operator !=(const YUV411& p)	const	{return !(*this == p);}
    
    element_type	x, y0, y1;
};

inline std::istream&
operator >>(std::istream& in, YUV411& yuv)
{
    return in >> (u_int&)yuv.y0 >> (u_int&)yuv.y1 >> (u_int&)yuv.x;
}

inline std::ostream&
operator <<(std::ostream& out, const YUV411& yuv)
{
    return out << u_int(yuv.y0) << ' ' << u_int(yuv.y1) << ' ' << u_int(yuv.x);
}

/************************************************************************
*  class ImageBase:	basic image class				*
************************************************************************/
//! 画素の2次元配列として定義されたあらゆる画像の基底となるクラス
class __PORT ImageBase
{
  public:
    typedef Matrix<double, 3, 4>	matrix34_type;
    
  //! 外部記憶に読み書きする際の画素のタイプ
    enum Type
    {
	DEFAULT = 0,	//!< same as internal type
	U_CHAR	= 5,	//!< unsigned mono	 8bit/pixel
	RGB_24	= 6,	//!< RGB		24bit/pixel	
	SHORT,		//!< signed mono	16bit/pixel
	INT,		//!< signed mono	32bit/pixel	
	FLOAT,		//!< float mono		32bit/pixel 
	DOUBLE,		//!< double mono	64bit/pixel
	YUV_444,	//!< YUV444		24bit/pixel
	YUV_422,	//!< YUV422		16bit/pixel
	YUYV_422,	//!< YUYV422		16bit/pixel
	YUV_411,	//!< YUV411		12bit/pixel
	BMP_8,		//!< BMP indexed color   8bit/pixel
	BMP_24,		//!< BMP BGR		24bit/pixel
	BMP_32		//!< BMP BGRA		32bit/pixel
    };

  //! 外部記憶に読み書きする際の付加情報
    struct TypeInfo
    {
	__PORT	TypeInfo(Type ty=DEFAULT)	;

	Type	type;		//!< 画素の型
	bool	bottomToTop;	//!< 行が下から上へ収められているならtrue
	size_t	ncolors;	//!< カラーパレットの色数
    };

  protected:
    template <class _T, class _DUMMY=void>
    struct type2type			{static constexpr Type value=RGB_24;};
    template <class _DUMMY>
    struct type2type<u_char, _DUMMY>	{static constexpr Type value=U_CHAR;};
    template <class _DUMMY>
    struct type2type<short, _DUMMY>	{static constexpr Type value=SHORT;};
    template <class _DUMMY>
    struct type2type<int, _DUMMY>	{static constexpr Type value=INT;};
    template <class _DUMMY>
    struct type2type<float, _DUMMY>	{static constexpr Type value=FLOAT;};
    template <class _DUMMY>
    struct type2type<double, _DUMMY>	{static constexpr Type value=DOUBLE;};
    template <class _DUMMY>
    struct type2type<YUV444, _DUMMY>	{static constexpr Type value=YUV_444;};
    template <class _DUMMY>
    struct type2type<YUV422, _DUMMY>	{static constexpr Type value=YUV_422;};
    template <class _DUMMY>
    struct type2type<YUYV422, _DUMMY>	{static constexpr Type value=YUYV_422;};
    template <class _DUMMY>
    struct type2type<YUV411, _DUMMY>	{static constexpr Type value=YUV_411;};
    
  protected:
  //! 画像を生成し投影行列と放射歪曲係数を初期化する．
  /*!
    投影行列は
    \f$\TUbeginarray{cc} \TUvec{I}{3\times 3} & \TUvec{0}{} \TUendarray\f$に，
    2つの放射歪曲係数はいずれも0に初期化される．
  */
    ImageBase()
	:P(), d1(0), d2(0)		{P[0][0] = P[1][1] = P[2][2] = 1.0;}
    virtual ~ImageBase()		;

    size_t		type2nbytes(Type type, bool padding)	const	;
    static size_t	type2depth(Type type)				;
    
  public:
    TypeInfo		restoreHeader(std::istream& in)			;
    Type		saveHeader(std::ostream& out,
				   Type type=DEFAULT)		const	;

  //! 画像の幅を返す．
  /*!
    \return	画像の幅
  */
    size_t		width()			const	{return _width();}

  //! 画像の高さを返す．
  /*!
    \return	画像の高さ
  */
    size_t		height()		const	{return _height();}

    size_t		npixelsToBorder(size_t u, size_t v,
					size_t dir)	const	;
    
  private:
    TypeInfo		restorePBMHeader(std::istream& in)	;
    TypeInfo		restoreBMPHeader(std::istream& in)	;
    Type		savePBMHeader(std::ostream& out,
				      Type type)	const	;
    Type		saveBMPHeader(std::ostream& out,
				      Type type)	const	;
    virtual size_t	_width()			const	= 0;
    virtual size_t	_height()			const	= 0;
    virtual Type	_defaultType()			const	= 0;
    virtual void	_resize(size_t h, size_t w,
				const TypeInfo& typeInfo)	= 0;

  public:
    matrix34_type	P;	//!< カメラの3x4投影行列
    double		d1;	//!< 放射歪曲の第1係数
    double		d2;	//!< 放射歪曲の第2係数
};

//! 指定された向きに沿った与えられた点から画像境界までの画素数を返す．
/*!
  \param u	始点の横座標
  \param v	始点の縦座標
  \param dir	8隣接方向
  \return	画像境界までの画素数(始点を含む)
*/
inline size_t
ImageBase::npixelsToBorder(size_t u, size_t v, size_t dir) const
{
    switch (dir % 8)
    {
      case 0:
	return width() - u;
      case 1:
	return std::min(width() - u, height() - v);
      case 2:
	return height() - v;
      case 3:
	return std::min(u + 1, height() - v);
      case 4:
	return u;
      case 5:
	return std::min(u + 1, v + 1);
      case 6:
	return v + 1;
    }

    return std::min(width() - u, v + 1);
}

/************************************************************************
*  class ImageLine<T, ALLOC>:	Generic image scanline class		*
************************************************************************/
//! T型の画素を持つ画像のスキャンラインを表すクラス
/*!
  \param T	画素の型
  \param ALLOC	アロケータの型
*/
template <class T, class ALLOC=std::allocator<T> >
class ImageLine : public Array<T, 0, ALLOC>
{
  private:
    typedef Array<T, 0, ALLOC>			super;

  public:
    typedef typename super::element_type	element_type;
    typedef typename super::pointer		pointer;
    
  public:
  //! 指定した画素数のスキャンラインを生成する．
  /*!
    \param d	画素数
  */
    explicit ImageLine(size_t d=0)	:super(d)			{}

  //! 外部の領域と画素数を指定してスキャンラインを生成する．
  /*!
    \param p	外部領域へのポインタ
    \param d	画素数
  */
    ImageLine(pointer p, size_t d)	:super(p, d)			{}

  //! 指定されたスキャンラインの部分スキャンラインを生成する．
  /*!
    \param l	元のスキャンライン
    \param u	部分スキャンラインの左端の座標
    \param d	部分スキャンラインの画素数
  */
    ImageLine(ImageLine& l, size_t u, size_t d) :super(l, u, d)		{}

#if !defined(__NVCC__)
  //! 他の配列と同一要素を持つスキャンラインを作る（コピーコンストラクタの拡張）．
  /*!
    コピーコンストラクタは別個自動的に生成される．
    \param expr	コピー元の配列
  */
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    ImageLine(const E& expr)	:super(expr)				{}

  //! 他の配列を自分に代入する（標準代入演算子の拡張）．
  /*!
    標準代入演算子は別個自動的に生成される．
    \param expr	コピー元の配列
    \return	この配列
  */
    template <class E>
    typename std::enable_if<is_range<E>::value, ImageLine&>::type
			operator =(const E& expr)
			{
			    super::operator =(expr);
			    return *this;
			}
#endif	// !__NVCC__
    
    ImageLine&		operator =(const element_type& c)		;
    const ImageLine	operator ()(size_t u, size_t d)		const	;
    ImageLine		operator ()(size_t u, size_t d)			;
    
    using		super::size;
    using		super::data;
    using		super::begin;
    using		super::end;

    template <class S>
    T			at(S uf)				const	;
    const YUV422*	copy(const YUV422* src)				;
    const YUYV422*	copy(const YUYV422* src)			;
    const YUV411*	copy(const YUV411* src)				;
    template <class ITER>
    ITER		copy(ITER src)					;
    template <class ITER, class TBL>
    ITER		lookup(ITER src, TBL tbl)			;
};

template <class T, class ALLOC> inline ImageLine<T, ALLOC>&
ImageLine<T, ALLOC>::operator =(const element_type& c)
{
    super::operator =(c);
    return *this;
}

//! このスキャンラインの部分スキャンラインを生成する．
/*!
  \param u	部分スキャンラインの左端の座標
  \param d	部分スキャンラインの画素数
  \return	生成された部分スキャンライン
*/
template <class T, class ALLOC> inline const ImageLine<T, ALLOC>
ImageLine<T, ALLOC>::operator ()(size_t u, size_t d) const
{
    return ImageLine(const_cast<ImageLine&>(*this), u, d);
}

//! このスキャンラインの部分スキャンラインを生成する．
/*!
  \param u	部分スキャンラインの左端の座標
  \param d	部分スキャンラインの画素数
  \return	生成された部分スキャンライン
*/
template <class T, class ALLOC> inline ImageLine<T, ALLOC>
ImageLine<T, ALLOC>::operator ()(size_t u, size_t d)
{
    return ImageLine(*this, u, d);
}
    
//! サブピクセル位置の画素値を線形補間で求める．
/*!
  指定された位置の両側の画素値を線形補間して出力する．
  \param uf	サブピクセルで指定された位置
  \return	線形補間された画素値
*/
template <class T, class ALLOC> template <class S> inline T
ImageLine<T, ALLOC>::at(S uf) const
{
    const int	u  = floor(uf);
    const T*	in = data() + u;
    const float	du = uf - u;
    return (du ? (1.0f - du) * *in + du * *(in + 1) : *in);
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T, class ALLOC> const YUV422*
ImageLine<T, ALLOC>::copy(const YUV422* src)
{
    for (auto dst = begin(); dst < end() - 1; )
    {
	*dst++ = YUV444(src[0].y, src[0].x, src[1].x);
	*dst++ = YUV444(src[1].y, src[0].x, src[1].x);
	src += 2;
    }
    return src;
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T, class ALLOC> const YUYV422*
ImageLine<T, ALLOC>::copy(const YUYV422* src)
{
    for (auto dst = begin(); dst < end() - 1; )
    {
	*dst++ = YUV444(src[0].y, src[0].x, src[1].x);
	*dst++ = YUV444(src[1].y, src[0].x, src[1].x);
	src += 2;
    }
    return src;
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T, class ALLOC> const YUV411*
ImageLine<T, ALLOC>::copy(const YUV411* src)
{
    for (auto dst = begin(); dst < end() - 3; )
    {
	*dst++ = YUV444(src[0].y0, src[0].x, src[1].x);
	*dst++ = YUV444(src[0].y1, src[0].x, src[1].x);
	*dst++ = YUV444(src[1].y0, src[0].x, src[1].x);
	*dst++ = YUV444(src[1].y1, src[0].x, src[1].x);
	src += 2;
    }
    return src;
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T, class ALLOC> template <class ITER> ITER
ImageLine<T, ALLOC>::copy(ITER src)
{
    for (auto dst = begin(); dst != end(); ++dst, ++src)
	*dst = *src;
    return src;
}

//! インデックスを読み込み，ルックアップテーブルで変換する．
/*!
  \param src	読み込み元の先頭を指す反復子
  \param tbl	ルックアップテーブルの先頭を指す反復子
  \return	最後に読み込まれた画素の次の画素への反復子
*/
template <class T, class ALLOC> template <class ITER, class TBL> ITER 
ImageLine<T, ALLOC>::lookup(ITER src, TBL tbl)
{
    for (auto dst = begin(); dst != end(); ++dst, ++src)
	*dst = T(tbl[*src]);
    return src;
}

template <class ALLOC>
class ImageLine<YUV422, ALLOC> : public Array<YUV422, 0, ALLOC>
{
  private:
    typedef Array<YUV422, 0, ALLOC>	super;

  public:
    typedef typename super::pointer	pointer;
    
  public:
    explicit ImageLine(size_t d=0)		:super(d)		{}
    ImageLine(pointer p, size_t d)		:super(p, d)		{}
    ImageLine(ImageLine& l, size_t u, size_t d) :super(l, u, d)		{}
    ImageLine&		operator =(const YUV422& c)			;
    const ImageLine	operator ()(size_t u, size_t d)		const	;
    ImageLine		operator ()(size_t u, size_t d)			;

    using		super::data;
    using		super::size;
    using		super::begin;
    using		super::end;
    
    const YUV444*	copy(const YUV444* src)				;
    const YUV422*	copy(const YUV422* src)				;
    const YUV411*	copy(const YUV411* src)				;
    template <class ITER>
    ITER		copy(ITER src)					;
    template <class ITER, class TBL>
    ITER		lookup(ITER src, TBL tbl)			;
};

template <class ALLOC> inline ImageLine<YUV422, ALLOC>&
ImageLine<YUV422, ALLOC>::operator =(const YUV422& c)
{
    super::operator =(c);
    return *this;
}
    
template <class ALLOC> inline const ImageLine<YUV422, ALLOC>
ImageLine<YUV422, ALLOC>::operator ()(size_t u, size_t d) const
{
    return ImageLine(const_cast<ImageLine&>(*this), u, d);
}
    
template <class ALLOC> inline ImageLine<YUV422, ALLOC>
ImageLine<YUV422, ALLOC>::operator ()(size_t u, size_t d)
{
    return ImageLine<YUV422, ALLOC>(*this, u, d);
}
    
template <class ALLOC> inline const YUV422*
ImageLine<YUV422, ALLOC>::copy(const YUV422* src)
{
    memcpy(data(), src, size() * sizeof(YUV422));
    return src + size();
}

template <class ALLOC> template <class ITER> ITER
ImageLine<YUV422, ALLOC>::copy(ITER src)
{
    for (auto dst = begin(); dst < end(); ++dst, ++src)
	*dst = YUV422(*src);
    return src;
}

template <class ALLOC> template <class ITER, class TBL> ITER
ImageLine<YUV422, ALLOC>::lookup(ITER src, TBL tbl)
{
    for (auto dst = begin(); dst < end(); ++dst, ++src)
	*dst = YUV422(tbl[*src]);
    return src;
}

template <class ALLOC>
class ImageLine<YUYV422, ALLOC> : public Array<YUYV422, 0, ALLOC>
{
  private:
    typedef Array<YUYV422, 0, ALLOC>	super;

  public:
    typedef typename super::pointer	pointer;
    
  public:
    explicit ImageLine(size_t d=0)		:super(d)		{}
    ImageLine(pointer p, size_t d)		:super(p, d)		{}
    ImageLine(ImageLine& l, size_t u, size_t d)	:super(l, u, d)		{}
    ImageLine&		operator =(const YUYV422& c)			;
    const ImageLine	operator ()(size_t u, size_t d)		const	;
    ImageLine		operator ()(size_t u, size_t d)			;

    using		super::size;
    using		super::data;
    using		super::begin;
    using		super::end;
    
    const YUV444*	copy(const YUV444* src)				;
    const YUYV422*	copy(const YUYV422* src)			;
    const YUV411*	copy(const YUV411* src)				;
    template <class ITER>
    ITER		copy(ITER src)					;
    template <class ITER, class TBL>
    ITER		lookup(ITER src, TBL tbl)			;
};
    
template <class ALLOC> inline ImageLine<YUYV422, ALLOC>&
ImageLine<YUYV422, ALLOC>::operator =(const YUYV422& c)
{
    super::operator =(c);
    return *this;
}
    
template <class ALLOC> inline const ImageLine<YUYV422, ALLOC>
ImageLine<YUYV422, ALLOC>::operator ()(size_t u, size_t d) const
{
    return ImageLine(const_cast<ImageLine&>(*this), u, d);
}
    
template <class ALLOC> inline ImageLine<YUYV422, ALLOC>
ImageLine<YUYV422, ALLOC>::operator ()(size_t u, size_t d)
{
    return ImageLine(*this, u, d);
}
    
template <class ALLOC> inline const YUYV422*
ImageLine<YUYV422, ALLOC>::copy(const YUYV422* src)
{
    memcpy(data(), src, size() * sizeof(YUYV422));
    return src + size();
}

template <class ALLOC> template <class ITER> ITER
ImageLine<YUYV422, ALLOC>::copy(ITER src)
{
    for (auto dst = begin(); dst < end(); ++dst, ++src)
	*dst = YUYV422(*src);
    return src;
}

template <class ALLOC> template <class ITER, class TBL> ITER
ImageLine<YUYV422, ALLOC>::lookup(ITER src, TBL tbl)
{
    for (auto dst = begin(); dst < end(); ++dst, ++src)
	*dst = YUYV422(tbl[*src]);
    return src;
}

template <class ALLOC>
class ImageLine<YUV411, ALLOC> : public Array<YUV411, 0, ALLOC>
{
  private:
    typedef Array<YUV411, 0, ALLOC>	super;

  public:
    typedef typename super::pointer	pointer;
    
  public:
    explicit ImageLine(size_t d=0)		:super(d/2)		{}
    ImageLine(pointer p, size_t d)		:super(p, d/2)		{}
    ImageLine(ImageLine& l, size_t u, size_t d)	:super(l, u/2, d/2)	{}
    ImageLine&		operator =(const YUV411& c)			;
    const ImageLine	operator ()(size_t u, size_t d)		const	;
    ImageLine		operator ()(size_t u, size_t d)			;

    using		super::data;
    using		super::size;
    using		super::begin;
    using		super::end;
    
    const YUV444*	copy(const YUV444* src)				;
    const YUV422*	copy(const YUV422* src)				;
    const YUYV422*	copy(const YUYV422* src)			;
    const YUV411*	copy(const YUV411* src)				;
    template <class ITER>
    ITER		copy(ITER src)					;
    template <class ITER, class TBL>
    ITER		lookup(ITER src, TBL tbl)			;

    bool		resize(size_t d)				;
    void		resize(pointer p, size_t d)			;
};

template <class ALLOC> inline ImageLine<YUV411, ALLOC>&
ImageLine<YUV411, ALLOC>::operator =(const YUV411& c)
{
    super::operator =(c);
    return *this;
}
    
template <class ALLOC> inline const ImageLine<YUV411, ALLOC>
ImageLine<YUV411, ALLOC>::operator ()(size_t u, size_t d) const
{
    return ImageLine(const_cast<ImageLine&>(*this), u, d);
}
    
template <class ALLOC> inline ImageLine<YUV411, ALLOC>
ImageLine<YUV411, ALLOC>::operator ()(size_t u, size_t d)
{
    return ImageLine(*this, u, d);
}
    
template <class ALLOC> inline const YUV411*
ImageLine<YUV411, ALLOC>::copy(const YUV411* src)
{
    memcpy(data(), src, size() * sizeof(YUV411));
    return src + size();
}

template <class ALLOC> template <class ITER> ITER
ImageLine<YUV411, ALLOC>::copy(ITER src)
{
    for (auto dst = begin(); dst < end(); ++dst)
    {
	*dst = YUV411(*src);
	src += 2;
    }
    return src;
}

template <class ALLOC> template <class ITER, class TBL> ITER
ImageLine<YUV411, ALLOC>::lookup(ITER src, TBL tbl)
{
    for (auto dst = begin(); dst < end(); ++dst)
    {
	*dst = YUV411(tbl[*src]);
	src += 2;
    }
    return src;
}

template <class ALLOC> inline bool
ImageLine<YUV411, ALLOC>::resize(size_t d)
{
    return super::resize(d/2);
}

template <class ALLOC> inline void
ImageLine<YUV411, ALLOC>::resize(pointer p, size_t d)
{
    super::resize(p, d/2);
}

/************************************************************************
*  class Image<T, ALLOC>						*
************************************************************************/
//! T型の画素を持つ画像を表すクラス
/*!
  \param T	画素の型
  \param ALLOC	アロケータの型
*/
template <class T, class ALLOC=std::allocator<T> >
class Image : public Array2<ImageLine<T, ALLOC> >, public ImageBase
{
  private:
    typedef Array2<ImageLine<T, ALLOC> >		super;
    
  public:
    typedef typename super::element_type		element_type;
    typedef typename super::pointer			pointer;

  public:
  //! 幅と高さを指定して画像を生成する．
  /*!
    \param w	画像の幅
    \param h	画像の高さ
    \param unit	1行あたりのバイト数がこの値の倍数になる
  */
    explicit Image(size_t w=0, size_t h=0, size_t unit=1)
	:super(h, w, unit), ImageBase()				{}

  //! 外部の領域と幅および高さを指定して画像を生成する．
  /*!
    \param p	外部領域へのポインタ
    \param w	画像の幅
    \param h	画像の高さ
  */
    Image(pointer p, size_t w, size_t h)
	:super(p, h, w), ImageBase()				{}

  //! 指定された画像の部分画像を生成する．
  /*!
    \param i	元の画像
    \param u	部分画像の左上端の横座標
    \param v	部分画像の左上端の縦座標
    \param w	部分画像の幅
    \param h	部分画像の高さ
  */
    Image(Image& i, size_t u, size_t v, size_t w, size_t h)
	:super(i, v, u, h, w), ImageBase(i)			{}

    Image&		operator =(const element_type& c)	;
    const Image		operator ()(size_t u, size_t v,
				    size_t w, size_t h)	const	;
    Image		operator ()(size_t u, size_t v,
				    size_t w, size_t h)		;

#if !defined(__NVCC__)
  //! 他の配列と同一要素を持つ画像を作る（コピーコンストラクタの拡張）．
  /*!
    コピーコンストラクタを定義しないと自動的に作られてしまうので，
    このコンストラクタがあってもコピーコンストラクタを別個に定義
    しなければならない．
    \param expr	コピー元の配列
  */
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Image(const E& expr)
	:super(expr), ImageBase()				{}

  //! 他の配列を自分に代入する（標準代入演算子の拡張）．
  /*!
    標準代入演算子を定義しないと自動的に作られてしまうので，この代入演算子が
    あっても標準代入演算子を別個に定義しなければならない．
    \param expr		コピー元の配列
    \return		この配列
  */
    template <class E>
    typename std::enable_if<is_range<E>::value, Image&>::type
		operator =(const E& expr)
		{
		    super::operator =(expr);
		    return *this;
		}
#endif	// !__NVCC__
    
    template <class S>
    T		at(const Point2<S>& p)			const	;

  //! 指定された位置の画素にアクセスする．
  /*!
    \param p	画素の位置
    \return	指定された画素
  */
    template <class S>
    const T&	operator ()(const Point2<S>& p)
					const	{return (*this)[p[1]][p[0]];}

  //! 指定された位置の画素にアクセスする．
  /*!
    \param p	画素の位置
    \return	指定された画素
  */
    template <class S>
    T&		operator ()(const Point2<S>& p)	{return (*this)[p[1]][p[0]];}
    
    size_t	width()			const	{return super::ncol();}
    size_t	height()		const	{return super::nrow();}

    using	super::begin;
    using	super::cbegin;
    using	super::end;
    using	super::cend;
    using	super::rbegin;
    using	super::crbegin;
    using	super::rend;
    using	super::crend;

    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out,
			     Type type=DEFAULT)			const	;
    std::istream&	restoreData(std::istream& in,
				    const TypeInfo& typeInfo)		;
    std::ostream&	saveData(std::ostream& out,
				 Type type=DEFAULT)		const	;

  private:
    template <class S>
    std::istream&	restoreRows(std::istream& in,
				    const TypeInfo& typeInfo)		;
    template <class S, class L>
    std::istream&	restoreAndLookupRows(std::istream& in,
					     const TypeInfo& typeInfo)	;
    template <class D, class L>
    std::ostream&	saveRows(std::ostream& out, Type type)	const	;
    Type		defaultType()				const	;
    
    virtual size_t	_width()				const	;
    virtual size_t	_height()				const	;
    virtual Type	_defaultType()				const	;
    virtual void	_resize(size_t h, size_t w, const TypeInfo&)	;
};

template <class T, class ALLOC> inline Image<T, ALLOC>&
Image<T, ALLOC>::operator =(const element_type& c)
{
    super::operator =(c);
    return *this;
}
    
//! この画像の部分画像を生成する．
/*!
  \param u	部分画像の左上端の横座標
  \param v	部分画像の左上端の縦座標
  \param w	部分画像の幅
  \param h	部分画像の高さ
  \return	生成された部分画像
*/
template <class T, class ALLOC> inline const Image<T, ALLOC>
Image<T, ALLOC>::operator ()(size_t u, size_t v, size_t w, size_t h) const
{
    return Image(const_cast<Image&>(*this), u, v, w, h);
}
    
//! この画像の部分画像を生成する．
/*!
  \param u	部分画像の左上端の横座標
  \param v	部分画像の左上端の縦座標
  \param w	部分画像の幅
  \param h	部分画像の高さ
  \return	生成された部分画像
*/
template <class T, class ALLOC> inline Image<T, ALLOC>
Image<T, ALLOC>::operator ()(size_t u, size_t v, size_t w, size_t h)
{
    return Image(*this, u, v, w, h);
}
    
//! サブピクセル位置の画素値を双線形補間で求める．
/*!
  指定された位置を囲む4つの画素値を双線形補間して出力する．
  \param p	サブピクセルで指定された位置
  \return	双線形補間された画素値
*/
template <class T, class ALLOC> template <class S> inline T
Image<T, ALLOC>::at(const Point2<S>& p) const
{
    const int	v    = floor(p[1]);
    const T	out0 = (*this)[v].at(p[0]);
    const float	dv   = p[1] - v;
    return (dv ? (1.0f - dv)*out0 + dv*(*this)[v+1].at(p[0]) : out0);
}

//! 入力ストリームから画像を読み込む．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, class ALLOC> inline std::istream&
Image<T, ALLOC>::restore(std::istream& in)
{
    return restoreData(in, restoreHeader(in));
}

//! 指定した画素タイプで出力ストリームに画像を書き出す．
/*!
  \param out	出力ストリーム
  \param type	画素タイプ．ただし， #DEFAULT を指定した場合は，
		この画像オブジェクトの画素タイプで書き出される．
  \return	outで指定した出力ストリーム
*/
template <class T, class ALLOC> inline std::ostream&
Image<T, ALLOC>::save(std::ostream& out, Type type) const
{
    return saveData(out, saveHeader(out, type));
}

//! 入力ストリームから画像の画素データを読み込む．
/*!
  \param in		入力ストリーム
  \param typeInfo	ストリーム中のデータの画素タイプ
			(読み込み先の画像の画素タイプではない)
  \return		inで指定した入力ストリーム
*/
template <class T, class ALLOC> std::istream&
Image<T, ALLOC>::restoreData(std::istream& in, const TypeInfo& typeInfo)
{
    switch (typeInfo.type)
    {
      case DEFAULT:
	break;
      case U_CHAR:
	return restoreRows<u_char>(in, typeInfo);
      case SHORT:
	return restoreRows<short>(in, typeInfo);
      case INT:
	return restoreRows<int>(in, typeInfo);
      case FLOAT:
	return restoreRows<float>(in, typeInfo);
      case DOUBLE:
	return restoreRows<double>(in, typeInfo);
      case RGB_24:
	return restoreRows<RGB>(in, typeInfo);
      case YUV_444:
	return restoreRows<YUV444>(in, typeInfo);
      case YUV_422:
	return restoreRows<YUV422>(in, typeInfo);
      case YUYV_422:
	return restoreRows<YUYV422>(in, typeInfo);
      case YUV_411:
	return restoreRows<YUV411>(in, typeInfo);
      case BMP_8:
	return restoreAndLookupRows<u_char, BGRA>(in, typeInfo);
      case BMP_24:
	return restoreRows<BGR>(in, typeInfo);
      case BMP_32:
	return restoreRows<BGRA>(in, typeInfo);
      default:
	throw std::runtime_error("Image<T, ALLOC>::restoreData(): unknown pixel type!!");
    }
    return in;
}

//! 指定した画素タイプで出力ストリームに画像の画素データを書き出す．
/*!
  \param out	出力ストリーム
  \param type	画素タイプ．ただし， #DEFAULT を指定した場合は，
		この画像オブジェクトの画素タイプで書き出される．
  \return	outで指定した出力ストリーム
*/
template <class T, class ALLOC> std::ostream&
Image<T, ALLOC>::saveData(std::ostream& out, Type type) const
{
    if (type == DEFAULT)
	type = defaultType();

    switch (type)
    {
      case U_CHAR:
	return saveRows<u_char, RGB>(out, type);
      case SHORT:
	return saveRows<short, RGB>(out, type);
      case INT:
	return saveRows<int, RGB>(out, type);
      case FLOAT:
	return saveRows<float, RGB>(out, type);
      case DOUBLE:
	return saveRows<double, RGB>(out, type);
      case RGB_24:
	return saveRows<RGB, RGB>(out, type);
      case YUV_444:
	return saveRows<YUV444, RGB>(out, type);
      case YUV_422:
	return saveRows<YUV422, RGB>(out, type);
      case YUYV_422:
	return saveRows<YUYV422, RGB>(out, type);
      case YUV_411:
	return saveRows<YUV411, RGB>(out, type);
      case BMP_8:
	return saveRows<u_char, BGRA>(out, type);
      case BMP_24:
	return saveRows<BGR, BGRA>(out, type);
      case BMP_32:
	return saveRows<BGRA, BGRA>(out, type);
      default:
	throw std::runtime_error("Image<T, ALLOC>::saveData(): unknown pixel type!!");
    }
    return out;
}

template <class T, class ALLOC> template <class S> std::istream&
Image<T, ALLOC>::restoreRows(std::istream& in, const TypeInfo& typeInfo)
{
    const size_t	npads = type2nbytes(typeInfo.type, true);
    ImageLine<S>	buf(width());
    if (typeInfo.bottomToTop)
    {
	for (auto line = rbegin(); line != rend(); ++line)
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    line->copy(buf.cbegin());
	}
    }
    else
    {
	for (auto line = begin(); line != end(); ++line)
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    line->copy(buf.cbegin());
	}
    }

    return in;
}

template <class T, class ALLOC> template <class S, class L> std::istream&
Image<T, ALLOC>::restoreAndLookupRows(std::istream& in,
				      const TypeInfo& typeInfo)
{
    Array<L>	colormap(typeInfo.ncolors);
    colormap.restore(in);
	
    const size_t	npads = type2nbytes(typeInfo.type, true);
    ImageLine<S>	buf(width());
    if (typeInfo.bottomToTop)
    {
	for (auto line = rbegin(); line != rend(); ++line)    
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    line->lookup(buf.cbegin(), colormap.cbegin());
	}
    }
    else
    {
	for (auto line = begin(); line != end(); ++line)    
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    line->lookup(buf.cbegin(), colormap.cbegin());
	}
    }

    return in;
}

template <class T, class ALLOC> template <class D, class L> std::ostream&
Image<T, ALLOC>::saveRows(std::ostream& out, Type type) const
{
    TypeInfo	typeInfo(type);

    Array<L>	colormap(typeInfo.ncolors);
    for (size_t i = 0; i < colormap.size(); ++i)
	colormap[i] = i;
    colormap.save(out);
    
    Array<u_char>	pad(type2nbytes(type, true));
    ImageLine<D>	buf(width());
    if (typeInfo.bottomToTop)
    {
	for (auto line = crbegin(); line != crend(); ++line)
	{
	    buf.copy(line->cbegin());
	    if (!buf.save(out) || !pad.save(out))
		break;
	}
    }
    else
    {
	for (auto line = cbegin(); line != cend(); ++line)
	{
	    buf.copy(line->cbegin());
	    if (!buf.save(out) || !pad.save(out))
		break;
	}
    }

    return out;
}

template <class T, class ALLOC> size_t
Image<T, ALLOC>::_width() const
{
    return Image::width();		// Don't call ImageBase::width!
}

template <class T, class ALLOC> size_t
Image<T, ALLOC>::_height() const
{
    return Image::height();		// Don't call ImageBase::height!
}

template <class T, class ALLOC> ImageBase::Type
Image<T, ALLOC>::_defaultType() const
{
    return Image::defaultType();
}

template <class T, class ALLOC> inline ImageBase::Type
Image<T, ALLOC>::defaultType() const
{
    return ImageBase::type2type<T>::value;
}

template <class T, class ALLOC> void
Image<T, ALLOC>::_resize(size_t h, size_t w, const TypeInfo&)
{
    Image<T, ALLOC>::resize(h, w);	// Don't call ImageBase::resize!
}

/************************************************************************
*  class GenericImage							*
************************************************************************/
//! 画素の型を問わない総称画像クラス
/*!
  個々の行や画素にアクセスすることはできない．
*/
class GenericImage : public ImageBase
{
  private:
    typedef Array2<Array<u_char> >	array2_type;

  public:
    typedef array2_type::pointer	pointer;
    typedef array2_type::const_pointer	const_pointer;
    
  public:
  //! 総称画像を生成する．
    GenericImage() :_a(), _typeInfo(U_CHAR), _colormap(0)		{}

    pointer		data()						;
    const_pointer	data()					const	;
    const TypeInfo&	typeInfo()				const	;
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out)			const	;
    std::istream&	restoreData(std::istream& in)			;
    std::ostream&	saveData(std::ostream& out)		const	;
    
  private:
    virtual size_t	_width()				const	;
    virtual size_t	_height()				const	;
    virtual Type	_defaultType()				const	;
    virtual void	_resize(size_t h, size_t w,
				const TypeInfo& typeInfo)		;

  private:
    array2_type		_a;
    TypeInfo		_typeInfo;
    Array<BGRA>		_colormap;
};

inline GenericImage::pointer
GenericImage::data()
{
    return _a.data();
}
    
inline GenericImage::const_pointer
GenericImage::data() const
{
    return _a.data();
}
    
//! 現在保持している画像のタイプ情報を返す．
/*!
  \return	タイプ情報
*/
inline const ImageBase::TypeInfo&
GenericImage::typeInfo() const
{
    return _typeInfo;
}

//! 入力ストリームから画像を読み込む．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
inline std::istream&
GenericImage::restore(std::istream& in)
{
    restoreHeader(in);
    return restoreData(in);
}

//! 出力ストリームに画像を書き出す．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
inline std::ostream&
GenericImage::save(std::ostream& out) const
{
    saveHeader(out, _typeInfo.type);
    return saveData(out);
}

}
#endif	// !__TU_IMAGEPP_H
