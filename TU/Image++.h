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
#ifndef	__TUImagePP_h
#define	__TUImagePP_h

#include <string.h>
#include <boost/operators.hpp>
#include "TU/Geometry++.h"

namespace TU
{
namespace detail
{
/************************************************************************
*  struct RGB, BGR, RGBA, ARGB, ABGR, BGRA				*
************************************************************************/
struct RGB
{
    RGB(u_char rr, u_char gg, u_char bb)	:r(rr), g(gg), b(bb)	{}
    
    u_char r, g, b;
};

struct BGR
{
    BGR(u_char rr, u_char gg, u_char bb)	:b(bb), g(gg), r(rr)	{}
    
    u_char b, g, r;
};

struct RGBA
{
    RGBA(u_char rr, u_char gg, u_char bb, u_char aa=255)
	:r(rr), g(gg), b(bb), a(aa)					{}
    
    u_char r, g, b, a;
};

struct ABGR
{
    ABGR(u_char rr, u_char gg, u_char bb, u_char aa=255)
	:a(aa), b(bb), g(gg), r(rr)					{}
    
    u_char a, b, g, r;
};

struct ARGB
{
    ARGB(u_char rr, u_char gg, u_char bb, u_char aa=255)
	:a(aa), r(rr), g(gg), b(bb)					{}
    
    u_char a, r, g, b;
};

struct BGRA
{
    BGRA(u_char rr, u_char gg, u_char bb, u_char aa=255)
	:b(bb), g(gg), r(rr), a(aa)					{}
    
    u_char b, g, r, a;
};
}

/************************************************************************
*  struct RGBB<E>							*
************************************************************************/
struct YUV444;

template <class E>
struct RGB_ : public E, boost::additive<RGB_<E>,
			boost::multiplicative<RGB_<E>, float,
			boost::equality_comparable<RGB_<E> > > >
{
    typedef u_char	element_type;

    RGB_()					     :E(0, 0, 0)	{}
    RGB_(u_char rr, u_char gg, u_char bb)	     :E(rr, gg, bb)	{}
    RGB_(u_char rr, u_char gg, u_char bb, u_char aa) :E(rr, gg, bb, aa)	{}
    RGB_(const RGB_<detail::RGB>& p)		     :E(p.r, p.g, p.b)	{}
    RGB_(const RGB_<detail::BGR>& p)		     :E(p.r, p.g, p.b)	{}
    RGB_(const YUV444& p)						;
    template <class E1>
    RGB_(const RGB_<E1>& p)	:E(p.r, p.g, p.b, p.a)			{}
    template <class T>
    RGB_(const T& p)		:E(u_char(p), u_char(p), u_char(p))	{}

    using	E::r;
    using	E::g;
    using	E::b;
	   
		operator u_char() const	{return u_char(float(*this));}
		operator s_char() const	{return s_char(float(*this));}
		operator short()  const	{return short(float(*this));}
		operator int()	  const	{return int(float(*this));}
		operator float()  const	{return 0.3f*r + 0.59f*g + 0.11f*b;}
		operator double() const	{return double(float(*this));}
    
    RGB_&	operator +=(const RGB_& p)
		{
		    r += p.r;
		    g += p.g;
		    b += p.b;
		    return *this;
		}
    RGB_&	operator -=(const RGB_& p)
		{
		    r -= p.r;
		    g -= p.g;
		    b -= p.b;
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
    return out << (u_int)p.r << ' ' << (u_int)p.g << ' ' << (u_int)p.b
	       << (u_int)p.a;
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
    return out << (u_int)p.r << ' ' << (u_int)p.g << ' ' << (u_int)p.b;
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
    return out << (u_int)p.r << ' ' << (u_int)p.g << ' ' << (u_int)p.b;
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
struct YUYV422;

//! Y, U, V（各8bit）の順で並んだカラー画素
struct YUV444
{
    YUV444(u_char yy=0, u_char uu=128, u_char vv=128)
			:u(uu), y(yy), v(vv)		{}
    template <class T> 
    YUV444(const T& p)	:u(128), y(u_char(p)), v(128)	{}

		operator u_char()		const	{return u_char(y);}
		operator s_char()		const	{return s_char(y);}
		operator short()		const	{return short(y);}
		operator int()			const	{return int(y);}
		operator float()		const	{return float(y);}
		operator double()		const	{return double(y);}
    bool	operator ==(const YUV444& yuv)	const	{return (u == yuv.u &&
								 y == yuv.y &&
								 v == yuv.v);}
    bool	operator !=(const YUV444& yuv)	const	{return !(*this==yuv);}
    

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

//! [U, Y0], [V, Y1]（各8bit）の順で並んだカラー画素(16bits/pixel)
struct YUV422
{
    YUV422(u_char yy=0, u_char xx=128)	:x(xx), y(yy)	{}
    YUV422(const YUYV422& p)				;
    template <class T>
    YUV422(const T& p)		:x(128), y(u_char(p))	{}

		operator u_char()		const	{return u_char(y);}
		operator s_char()		const	{return s_char(y);}
		operator short()		const	{return short(y);}
		operator int()			const	{return int(y);}
		operator float()		const	{return float(y);}
		operator double()		const	{return double(y);}
    bool	operator ==(const YUV422& p)	const	{return (x == p.x &&
								 y == p.y);}
    bool	operator !=(const YUV422& p)	const	{return !(*this == p);}
    
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

//! [Y0, U], [Y1, V]（各8bit）の順で並んだカラー画素(16bits/pixel)
struct YUYV422
{
    YUYV422(u_char yy=0, u_char xx=128)	:y(yy), x(xx)	{}
    YUYV422(const YUV422& p)				;
    template <class T>
    YUYV422(const T& p)		:y(u_char(p)), x(128) 	{}

		operator u_char()		const	{return u_char(y);}
		operator s_char()		const	{return s_char(y);}
		operator short()		const	{return short(y);}
		operator int()			const	{return int(y);}
		operator float()		const	{return float(y);}
		operator double()		const	{return double(y);}
    bool	operator ==(const YUYV422& p)	const	{return (y == p.y &&
								 x == p.x);}
    bool	operator !=(const YUYV422& p)	const	{return !(*this == p);}
    
    u_char	y, x;
};

inline std::istream&
operator >>(std::istream& in, YUYV422& yuv)
{
    return in >> (u_int&)yuv.y >> (u_int&)yuv.x;
}

inline std::ostream&
operator <<(std::ostream& out, const YUYV422& yuv)
{
    return out << (u_int)yuv.y << ' ' << (u_int)yuv.x;
}

//! [U, Y0, Y1], [V, Y2, Y3]（各8bit）の順で並んだカラー画素(12bits/pixel)
struct YUV411
{
    YUV411(u_char yy0=0, u_char yy1=0, u_char xx=128)
			:x(xx), y0(yy0), y1(yy1)			{}
    template <class T>
    YUV411(const T& p)	:x(128), y0(u_char(p)), y1(u_char(*(&p+1)))	{}

    bool	operator ==(const YUV411& p)	const	{return (x  == p.x  &&
								 y0 == p.y0 &&
								 y1 == p.y1);}
    bool	operator !=(const YUV411& p)	const	{return !(*this == p);}
    
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

inline
YUV422::YUV422(const YUYV422& p)  :x(p.x), y(p.y)	{}

inline
YUYV422::YUYV422(const YUV422& p) :y(p.y), x(p.x)	{}

/************************************************************************
*  function fromYUV<T>()						*
************************************************************************/
//! カラーのY, U, V値を与えて他のカラー表現に変換するクラス
class __PORT ConversionFromYUV
{
  public:
    ConversionFromYUV()					;

  private:
    template <class T>
    friend T	fromYUV(u_char y, u_char u, u_char v)	;
    
    int		_r[256], _g0[256], _g1[256], _b[256];
};

__PORT extern const ConversionFromYUV	conversionFromYUV;

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

template <> inline s_char
fromYUV<s_char>(u_char y, u_char, u_char)
{
    return y;
}

template <> inline short
fromYUV<short>(u_char y, u_char, u_char)
{
    return y;
}

template <> inline int
fromYUV<int>(u_char y, u_char, u_char)
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

template <class E> inline
RGB_<E>::RGB_(const YUV444& p)
    :E(fromYUV<RGB_>(p.y, p.u, p.v))
{
}

/************************************************************************
*  class ImageBase:	basic image class				*
************************************************************************/
//! 画素の2次元配列として定義されたあらゆる画像の基底となるクラス
class __PORT ImageBase
{
  public:
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
	u_int	ncolors;	//!< カラーパレットの色数
    };
    
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

    u_int		type2nbytes(Type type, bool padding)	const	;
    static u_int	type2depth(Type type)				;
    
  public:
    TypeInfo		restoreHeader(std::istream& in)			;
    Type		saveHeader(std::ostream& out,
				   Type type=DEFAULT)		const	;

  //! 画像の幅を返す．
  /*!
    \return	画像の幅
  */
    u_int		width()			const	{return _width();}

  //! 画像の高さを返す．
  /*!
    \return	画像の高さ
  */
    u_int		height()		const	{return _height();}

  //! 画像のサイズを変更する．
  /*!
    \param h	新しい幅
    \param w	新しい高さ
  */
    void		resize(u_int h, u_int w)	{_resize(h, w,
								 DEFAULT);}
	
  private:
    TypeInfo		restorePBMHeader(std::istream& in)	;
    TypeInfo		restoreBMPHeader(std::istream& in)	;
    Type		savePBMHeader(std::ostream& out,
				      Type type)	const	;
    Type		saveBMPHeader(std::ostream& out,
				      Type type)	const	;
    virtual u_int	_width()			const	= 0;
    virtual u_int	_height()			const	= 0;
    virtual Type	_defaultType()			const	= 0;
    virtual void	_resize(u_int h, u_int w,
				const TypeInfo& typeInfo)	= 0;

  public:
    Matrix34d		P;			//!< カメラの3x4投影行列
    double		d1;			//!< 放射歪曲の第1係数
    double		d2;			//!< 放射歪曲の第2係数
};

/************************************************************************
*  class ImageLine<T>:	Generic image scanline class			*
************************************************************************/
//! T型の画素を持つ画像のスキャンラインを表すクラス
/*!
  \param T	画素の型
*/
template <class T>
class ImageLine : public Array<T>
{
  private:
    typedef Array<T>					super;

  public:
    typedef typename super::element_type		element_type;
    typedef typename super::value_type			value_type;
    typedef typename super::difference_type		difference_type;
    typedef typename super::reference			reference;
    typedef typename super::const_reference		const_reference;
    typedef typename super::pointer			pointer;
    typedef typename super::const_pointer		const_pointer;
    typedef typename super::iterator			iterator;
    typedef typename super::const_iterator		const_iterator;
    typedef typename super::reverse_iterator		reverse_iterator;
    typedef typename super::const_reverse_iterator	const_reverse_iterator;
    
  public:
  //! 指定した画素数のスキャンラインを生成する．
  /*!
    \param d	画素数
  */
    explicit ImageLine(u_int d=0)
        :super(d), _lmost(0), _rmost(d)				{*this = 0;}

  //! 外部の領域と画素数を指定してスキャンラインを生成する．
  /*!
    \param p	外部領域へのポインタ
    \param d	画素数
  */
    ImageLine(T* p, u_int d)
        :super(p, d), _lmost(0), _rmost(d)			{}

  //! 指定されたスキャンラインの部分スキャンラインを生成する．
  /*!
    \param l	元のスキャンライン
    \param u	部分スキャンラインの左端の座標
    \param d	部分スキャンラインの画素数
  */
    ImageLine(ImageLine<T>& l, u_int u, u_int d)
	:super(l, u, d), _lmost(0), _rmost(d)			{}

    const ImageLine	operator ()(u_int u, u_int d)	const	;
    ImageLine		operator ()(u_int u, u_int d)		;
    
  //! 全ての画素に同一の値を代入する．
  /*!
    \param c	代入する画素値
    \return	このスキャンライン
  */
    ImageLine&		operator =(const T& c)		{super::operator =(c);
							 return *this;}

    using		super::begin;
    using		super::end;
    using		super::rbegin;
    using		super::rend;
    using		super::size;
    using		super::ptr;

    template <class S>
    T			at(S uf)		const	;
    const YUV422*	fill(const YUV422* src)		;
    const YUYV422*	fill(const YUYV422* src)	;
    const YUV411*	fill(const YUV411* src)		;
    const T*		fill(const T* src)		;
    template <class S>
    const S*		fill(const S* src)		;
    template <class S, class L>
    const S*		lookup(const S* src,
			       const L* tbl)		;

  //! 左端の有効画素の位置を返す．
  /*!
    \return	左端の有効画素の位置
  */
    int			lmost()			const	{return _lmost;}

  //! 右端の有効画素の次の位置を返す．
  /*!
    \return	右端の有効画素の次の位置
  */
    int			rmost()			const	{return _rmost;}

  //! 有効画素の範囲を設定する．
  /*!
    \param l	有効画素の左端
    \param r	有効画素の右端の次
  */
    void		setLimits(int l, int r)		{_lmost = l;
							 _rmost = r;}
  //! 指定された位置の画素が有効か判定する．
  /*!
    \param u	画素の位置
    \return	有効ならばtrue，無効ならばfalse
  */
    bool		valid(int u)		const	{return (u >= _lmost &&
								 u <  _rmost);}
	
    bool		resize(u_int d)			;
    void		resize(T* p, u_int d)		;

  private:
    int			_lmost;
    int			_rmost;
};

//! このスキャンラインの部分スキャンラインを生成する．
/*!
  \param u	部分スキャンラインの左端の座標
  \param d	部分スキャンラインの画素数
  \return	生成された部分スキャンライン
*/
template <class T> inline const ImageLine<T>
ImageLine<T>::operator ()(u_int u, u_int d) const
{
    return ImageLine<T>(const_cast<ImageLine<T>&>(*this), u, d);
}

//! このスキャンラインの部分スキャンラインを生成する．
/*!
  \param u	部分スキャンラインの左端の座標
  \param d	部分スキャンラインの画素数
  \return	生成された部分スキャンライン
*/
template <class T> inline ImageLine<T>
ImageLine<T>::operator ()(u_int u, u_int d)
{
    return ImageLine<T>(*this, u, d);
}
    
//! サブピクセル位置の画素値を線形補間で求める．
/*!
  指定された位置の両側の画素値を線形補間して出力する．
  \param uf	サブピクセルで指定された位置
  \return	線形補間された画素値
*/
template <class T> template <class S> inline T
ImageLine<T>::at(S uf) const
{
    const int	u  = floor(uf);
    const T*	in = ptr() + u;
    const float	du = uf - u;
    return (du ? (1.0f - du) * *in + du * *(in + 1) : *in);
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T> const YUV422*
ImageLine<T>::fill(const YUV422* src)
{
    for (iterator dst = begin(); dst < end() - 1; )
    {
	*dst++ = fromYUV<T>(src[0].y, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[1].y, src[0].x, src[1].x);
	src += 2;
    }
    return src;
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T> const YUYV422*
ImageLine<T>::fill(const YUYV422* src)
{
    for (iterator dst = begin(); dst < end() - 1; )
    {
	*dst++ = fromYUV<T>(src[0].y, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[1].y, src[0].x, src[1].x);
	src += 2;
    }
    return src;
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T> const YUV411*
ImageLine<T>::fill(const YUV411* src)
{
    for (iterator dst = begin(); dst < end() - 3; )
    {
	*dst++ = fromYUV<T>(src[0].y0, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[0].y1, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[1].y0, src[0].x, src[1].x);
	*dst++ = fromYUV<T>(src[1].y1, src[0].x, src[1].x);
	src += 2;
    }
    return src;
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T> template <class S> const S*
ImageLine<T>::fill(const S* src)
{
    for (iterator dst = begin(); dst != end(); )
	*dst++ = T(*src++);
    return src;
}

//! インデックスを読み込み，ルックアップテーブルで変換する．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \param tbl	ルックアップテーブルの先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T> template <class S, class L> const S*
ImageLine<T>::lookup(const S* src, const L* tbl)
{
    for (iterator dst = begin(); dst != end(); )
	*dst++ = T(*(tbl + *src++));
    return src;
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T> inline const T*
ImageLine<T>::fill(const T* src)
{
    memcpy(ptr(), src, size() * sizeof(T));
    return src + size();
}

//! スキャンラインの画素数を変更する．
/*!
  ただし，他のオブジェクトと記憶領域を共有しているスキャンラインの画素数を
  変更することはできない．
  \param d			新しい画素数
  \return			dが元の画素数よりも大きければtrue，そう
				でなければfalse
  \throw std::logic_error	記憶領域を他のオブジェクトと共有している場合
				に送出
*/
template <class T> inline bool
ImageLine<T>::resize(u_int d)
{
    _lmost = 0;
    _rmost = d;
    return super::resize(d);
}

//! スキャンラインが内部で使用する記憶領域を指定したものに変更する．
/*!
  \param p	新しい記憶領域へのポインタ
  \param d	新しい画素数
*/
template <class T> inline void
ImageLine<T>::resize(T* p, u_int d)
{
    _lmost = 0;
    _rmost = d;
    super::resize(p, d);
}

template <>
class ImageLine<YUV422> : public Array<YUV422>
{
  private:
    typedef Array<YUV422>			super;

  public:
    explicit ImageLine(u_int d=0)
	:super(d), _lmost(0), _rmost(d)				{}
    ImageLine(YUV422* p, u_int d)
	:super(p, d), _lmost(0), _rmost(d)			{}
    ImageLine(ImageLine<YUV422>& l, u_int u, u_int d)
	:super(l, u, d), _lmost(0), _rmost(d)			{}
    const ImageLine	operator ()(u_int u, u_int d)	const	;
    ImageLine		operator ()(u_int u, u_int d)		;
    ImageLine&		operator =(YUV422 c)		{super::operator =(c);
							 return *this;}

    const YUV444*	fill(const YUV444* src)		;
    const YUV422*	fill(const YUV422* src)		;
    const YUV411*	fill(const YUV411* src)		;
    template <class S>
    const S*		fill(const S* src)		;
    template <class S, class L>
    const S*		lookup(const S* src,
			       const L* tbl)		;
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

inline const ImageLine<YUV422>
ImageLine<YUV422>::operator ()(u_int u, u_int d) const
{
    return ImageLine<YUV422>(const_cast<ImageLine<YUV422>&>(*this), u, d);
}
    
inline ImageLine<YUV422>
ImageLine<YUV422>::operator ()(u_int u, u_int d)
{
    return ImageLine<YUV422>(*this, u, d);
}
    
inline const YUV422*
ImageLine<YUV422>::fill(const YUV422* src)
{
    memcpy(ptr(), src, size() * sizeof(YUV422));
    return src + size();
}

template <class S> const S*
ImageLine<YUV422>::fill(const S* src)
{
    for (iterator dst = begin(); dst < end(); )
	*dst++ = YUV422(*src++);
    return src;
}

template <class S, class L> const S*
ImageLine<YUV422>::lookup(const S* src, const L* tbl)
{
    for (iterator dst = begin(); dst < end(); )
	*dst++ = YUV422(*(tbl + *src++));
    return src;
}

inline bool
ImageLine<YUV422>::resize(u_int d)
{
    _lmost = 0;
    _rmost = d;
    return super::resize(d);
}

inline void
ImageLine<YUV422>::resize(YUV422* p, u_int d)
{
    _lmost = 0;
    _rmost = d;
    super::resize(p, d);
}

template <>
class ImageLine<YUYV422> : public Array<YUYV422>
{
  private:
    typedef Array<YUYV422>			super;

  public:
    explicit ImageLine(u_int d=0)
	:super(d), _lmost(0), _rmost(d)				{}
    ImageLine(YUYV422* p, u_int d)
	:super(p, d), _lmost(0), _rmost(d)			{}
    ImageLine(ImageLine<YUYV422>& l, u_int u, u_int d)
	:super(l, u, d), _lmost(0), _rmost(d)			{}
    const ImageLine	operator ()(u_int u, u_int d)	const	;
    ImageLine		operator ()(u_int u, u_int d)		;
    ImageLine&		operator =(YUYV422 c)		{super::operator =(c);
							 return *this;}

    const YUV444*	fill(const YUV444* src)		;
    const YUYV422*	fill(const YUYV422* src)	;
    const YUV411*	fill(const YUV411* src)		;
    template <class S>
    const S*		fill(const S* src)		;
    template <class S, class L>
    const S*		lookup(const S* src,
			       const L* tbl)		;
    int			lmost()			const	{return _lmost;}
    int			rmost()			const	{return _rmost;}
    void		setLimits(int l, int r)		{_lmost = l;
							 _rmost = r;}
    bool		valid(int u)		const	{return (u >= _lmost &&
								 u <  _rmost);}
	
    bool		resize(u_int d)			;
    void		resize(YUYV422* p, u_int d)	;

  private:
    int			_lmost;
    int			_rmost;
};

inline const ImageLine<YUYV422>
ImageLine<YUYV422>::operator ()(u_int u, u_int d) const
{
    return ImageLine<YUYV422>(const_cast<ImageLine<YUYV422>&>(*this), u, d);
}
    
inline ImageLine<YUYV422>
ImageLine<YUYV422>::operator ()(u_int u, u_int d)
{
    return ImageLine<YUYV422>(*this, u, d);
}
    
inline const YUYV422*
ImageLine<YUYV422>::fill(const YUYV422* src)
{
    memcpy(ptr(), src, size() * sizeof(YUYV422));
    return src + size();
}

template <class S> const S*
ImageLine<YUYV422>::fill(const S* src)
{
    for (iterator dst = begin(); dst < end(); )
	*dst++ = YUYV422(*src++);
    return src;
}

template <class S, class L> const S*
ImageLine<YUYV422>::lookup(const S* src, const L* tbl)
{
    for (iterator dst = begin(); dst < end(); )
	*dst++ = YUYV422(*(tbl + *src++));
    return src;
}

inline bool
ImageLine<YUYV422>::resize(u_int d)
{
    _lmost = 0;
    _rmost = d;
    return super::resize(d);
}

inline void
ImageLine<YUYV422>::resize(YUYV422* p, u_int d)
{
    _lmost = 0;
    _rmost = d;
    super::resize(p, d);
}

template <>
class ImageLine<YUV411> : public Array<YUV411>
{
  private:
    typedef Array<YUV411>			super;

  public:
    explicit ImageLine(u_int d=0)
	:super(d/2), _lmost(0), _rmost(d)			{}
    ImageLine(YUV411* p, u_int d)
	:super(p, d/2), _lmost(0), _rmost(d)			{}
    ImageLine(ImageLine<YUV411>& l, u_int u, u_int d)
	:super(l, u/2, d/2), _lmost(0), _rmost(d)		{}
    const ImageLine	operator ()(u_int u, u_int d)	const	;
    ImageLine		operator ()(u_int u, u_int d)		;
    ImageLine&		operator =(YUV411 c)		{super::operator =(c);
							 return *this;}
    
    const YUV444*	fill(const YUV444* src)		;
    const YUV422*	fill(const YUV422* src)		;
    const YUYV422*	fill(const YUYV422* src)	;
    const YUV411*	fill(const YUV411* src)		;
    template <class S>
    const S*		fill(const S* src)		;
    template <class S, class L>
    const S*		lookup(const S* src,
			       const L* tbl)		;
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

inline const ImageLine<YUV411>
ImageLine<YUV411>::operator ()(u_int u, u_int d) const
{
    return ImageLine<YUV411>(const_cast<ImageLine<YUV411>&>(*this), u, d);
}
    
inline ImageLine<YUV411>
ImageLine<YUV411>::operator ()(u_int u, u_int d)
{
    return ImageLine<YUV411>(*this, u, d);
}
    
inline const YUV411*
ImageLine<YUV411>::fill(const YUV411* src)
{
    memcpy(ptr(), src, size() * sizeof(YUV411));
    return src + size();
}

template <class S> const S*
ImageLine<YUV411>::fill(const S* src)
{
    for (iterator dst = begin(); dst < end(); )
    {
	*dst++ = YUV411(*src);
	src += 2;
    }
    return src;
}

template <class S, class L> const S*
ImageLine<YUV411>::lookup(const S* src, const L* tbl)
{
    for (iterator dst = begin(); dst < end(); )
    {
	*dst++ = YUV411(*(tbl + *src));
	src += 2;
    }
    return src;
}

inline bool
ImageLine<YUV411>::resize(u_int d)
{
    _lmost = 0;
    _rmost = d;
    return super::resize(d/2);
}

inline void
ImageLine<YUV411>::resize(YUV411* p, u_int d)
{
    _lmost = 0;
    _rmost = d;
    super::resize(p, d/2);
}

/************************************************************************
*  class Image<T>:							*
************************************************************************/
//! T型の画素を持つ画像を表すクラス
/*!
  \param T	画素の型
  \param B	バッファの型
*/
template <class T, class B=Buf<T> >
class Image : public Array2<ImageLine<T>, B>, public ImageBase
{
  private:
    typedef Array2<ImageLine<T>, B>			super;
    
  public:
    typedef typename super::element_type		element_type;
    typedef typename super::value_type			value_type;
    typedef typename super::difference_type		difference_type;
    typedef typename super::reference			reference;
    typedef typename super::const_reference		const_reference;
    typedef typename super::pointer			pointer;
    typedef typename super::const_pointer		const_pointer;
    typedef typename super::iterator			iterator;
    typedef typename super::const_iterator		const_iterator;
    typedef typename super::reverse_iterator		reverse_iterator;
    typedef typename super::const_reverse_iterator	const_reverse_iterator;

  public:
  //! 幅と高さを指定して画像を生成する．
  /*!
    \param w	画像の幅
    \param h	画像の高さ
  */
    explicit Image(u_int w=0, u_int h=0)
	:super(h, w), ImageBase()				{*this = 0;}

  //! 外部の領域と幅および高さを指定して画像を生成する．
  /*!
    \param p	外部領域へのポインタ
    \param w	画像の幅
    \param h	画像の高さ
  */
    Image(T* p, u_int w, u_int h)
	:super(p, h, w), ImageBase()				{}

  //! 指定された画像の部分画像を生成する．
  /*!
    \param i	元の画像
    \param u	部分画像の左上端の横座標
    \param v	部分画像の左上端の縦座標
    \param w	部分画像の幅
    \param h	部分画像の高さ
  */
    template <class B2>
    Image(Image<T, B2>& i, u_int u, u_int v, u_int w, u_int h)
	:super(i, v, u, h, w), ImageBase(i)			{}

    const Image<T>	operator ()(u_int u, u_int v,
				    u_int w, u_int h)	const	;
    Image<T>		operator ()(u_int u, u_int v,
				    u_int w, u_int h)		;
    
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
    
    u_int	width()			const	{return super::ncol();}
    u_int	height()		const	{return super::nrow();}
    
  //! 全ての画素に同一の値を代入する．
  /*!
    \param c	代入する画素値
    \return	この画像
  */
    Image&	operator =(const T& c)		{super::operator =(c);
						 return *this;}

    using	super::begin;
    using	super::end;
    using	super::rbegin;
    using	super::rend;
    using	super::ptr;

    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out,
			     Type type=DEFAULT)			const	;
    std::istream&	restoreData(std::istream& in,
				    const TypeInfo& typeInfo)		;
    std::ostream&	saveData(std::ostream& out,
				 Type type=DEFAULT)		const	;
    void		resize(u_int h, u_int w)			;
    void		resize(T* p, u_int h, u_int w)			;

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
    
    virtual u_int	_width()				const	;
    virtual u_int	_height()				const	;
    virtual Type	_defaultType()				const	;
    virtual void	_resize(u_int h, u_int w, const TypeInfo&)	;
};

//! この画像の部分画像を生成する．
/*!
  \param u	部分画像の左上端の横座標
  \param v	部分画像の左上端の縦座標
  \param w	部分画像の幅
  \param h	部分画像の高さ
  \return	生成された部分画像
*/
template <class T, class B> inline const Image<T>
Image<T, B>::operator ()(u_int u, u_int v, u_int w, u_int h) const
{
    return Image<T>(const_cast<Image<T>&>(*this), u, v, w, h);
}
    
//! この画像の部分画像を生成する．
/*!
  \param u	部分画像の左上端の横座標
  \param v	部分画像の左上端の縦座標
  \param w	部分画像の幅
  \param h	部分画像の高さ
  \return	生成された部分画像
*/
template <class T, class B> inline Image<T>
Image<T, B>::operator ()(u_int u, u_int v, u_int w, u_int h)
{
    return Image<T>(*this, u, v, w, h);
}
    
//! サブピクセル位置の画素値を双線形補間で求める．
/*!
  指定された位置を囲む4つの画素値を双線形補間して出力する．
  \param p	サブピクセルで指定された位置
  \return	双線形補間された画素値
*/
template <class T, class B> template <class S> inline T
Image<T, B>::at(const Point2<S>& p) const
{
    const int	v    = floor(p[1]);
    const T	out0 = (*this)[v].at(p[0]);
    const float	dv   = p[1] - vf;
    return (dv ? (1.0f - dv)*out0 + dv*(*this)[v+1].at(p[0]) : out0);
}

//! 入力ストリームから画像を読み込む．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T, class B> inline std::istream&
Image<T, B>::restore(std::istream& in)
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
template <class T, class B> inline std::ostream&
Image<T, B>::save(std::ostream& out, Type type) const
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
template <class T, class B> std::istream&
Image<T, B>::restoreData(std::istream& in, const TypeInfo& typeInfo)
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
	throw std::runtime_error("Image<T, B>::restoreData(): unknown pixel type!!");
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
template <class T, class B> std::ostream&
Image<T, B>::saveData(std::ostream& out, Type type) const
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
	throw std::runtime_error("Image<T, B>::saveData(): unknown pixel type!!");
    }
    return out;
}

//! 画像のサイズを変更する．
/*!
  \param h	新しい高さ
  \param w	新しい幅
*/
template <class T, class B> inline void
Image<T, B>::resize(u_int h, u_int w)
{
    super::resize(h, w);
}

//! 外部の領域を指定して画像のサイズを変更する．
/*!
  \param p	外部領域へのポインタ
  \param h	画像の高さ
  \param w	画像の幅
  */
template <class T, class B> inline void
Image<T, B>::resize(T* p, u_int h, u_int w)
{
    super::resize(p, h, w);
}
 
template <class T, class B> template <class S> std::istream&
Image<T, B>::restoreRows(std::istream& in, const TypeInfo& typeInfo)
{
    const u_int		npads = type2nbytes(typeInfo.type, true);
    ImageLine<S>	buf(width());
    if (typeInfo.bottomToTop)
    {
	for (reverse_iterator line = rbegin(); line != rend(); ++line)
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    line->fill(buf.ptr());
	}
    }
    else
    {
	for (iterator line = begin(); line != end(); ++line)
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    line->fill(buf.ptr());
	}
    }

    return in;
}

template <class T, class B> template <class S, class L> std::istream&
Image<T, B>::restoreAndLookupRows(std::istream& in, const TypeInfo& typeInfo)
{
    Array<L>	colormap(typeInfo.ncolors);
    colormap.restore(in);
	
    const u_int		npads = type2nbytes(typeInfo.type, true);
    ImageLine<S>	buf(width());
    if (typeInfo.bottomToTop)
    {
	for (reverse_iterator line = rbegin(); line != rend(); ++line)    
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    line->lookup(buf.ptr(), colormap.ptr());
	}
    }
    else
    {
	for (iterator line = begin(); line != end(); ++line)    
	{
	    if (!buf.restore(in) || !in.ignore(npads))
		break;
	    line->lookup(buf.ptr(), colormap.ptr());
	}
    }

    return in;
}

template <class T, class B> template <class D, class L> std::ostream&
Image<T, B>::saveRows(std::ostream& out, Type type) const
{
    TypeInfo	typeInfo(type);

    Array<L>	colormap(typeInfo.ncolors);
    for (u_int i = 0; i < colormap.size(); ++i)
	colormap[i] = i;
    colormap.save(out);
    
    Array<u_char>	pad(type2nbytes(type, true));
    pad = 0;
    
    ImageLine<D>	buf(width());
    if (typeInfo.bottomToTop)
    {
	for (const_reverse_iterator line = rbegin(); line != rend(); ++line)
	{
	    buf.fill(line->ptr());
	    if (!buf.save(out) || !pad.save(out))
		break;
	}
    }
    else
    {
	for (const_iterator line = begin(); line != end(); ++line)
	{
	    buf.fill(line->ptr());
	    if (!buf.save(out) || !pad.save(out))
		break;
	}
    }

    return out;
}

template <class T, class B> u_int
Image<T, B>::_width() const
{
    return Image<T, B>::width();	// Don't call ImageBase::width!
}

template <class T, class B> u_int
Image<T, B>::_height() const
{
    return Image<T, B>::height();	// Don't call ImageBase::height!
}

template <class T, class B> ImageBase::Type
Image<T, B>::_defaultType() const
{
    return Image<T, B>::defaultType();
}

template <class T, class B> inline ImageBase::Type
Image<T, B>::defaultType() const
{
    return RGB_24;
}

template <> inline ImageBase::Type
Image<u_char, Buf<u_char> >::defaultType() const
{
    return U_CHAR;
}

template <> inline ImageBase::Type
Image<short, Buf<short> >::defaultType() const
{
    return SHORT;
}

template <> inline ImageBase::Type
Image<int, Buf<int> >::defaultType() const
{
    return INT;
}

template <> inline ImageBase::Type
Image<float, Buf<float> >::defaultType() const
{
    return FLOAT;
}

template <> inline ImageBase::Type
Image<double, Buf<double> >::defaultType() const
{
    return DOUBLE;
}

template <> inline ImageBase::Type
Image<YUV444, Buf<YUV444> >::defaultType() const
{
    return YUV_444;
}

template <> inline ImageBase::Type
Image<YUV422, Buf<YUV422> >::defaultType() const
{
    return YUV_422;
}

template <> inline ImageBase::Type
Image<YUYV422, Buf<YUYV422> >::defaultType() const
{
    return YUYV_422;
}

template <> inline ImageBase::Type
Image<YUV411, Buf<YUV411> >::defaultType() const
{
    return YUV_411;
}

template <class T, class B> void
Image<T, B>::_resize(u_int h, u_int w, const TypeInfo&)
{
    Image<T, B>::resize(h, w);		// Don't call ImageBase::resize!
}

template <> inline u_int
Buf<YUV411>::stride(u_int siz)
{
    return siz / 2;
}

/************************************************************************
*  class GenericImage							*
************************************************************************/
//! 画素の型を問わない総称画像クラス
/*!
  個々の行や画素にアクセスすることはできない．
*/
class GenericImage : public Array2<Array<u_char> >, public ImageBase
{
  public:
  //! 総称画像を生成する．
    GenericImage() :_typeInfo(U_CHAR), _colormap(0)			{}

    const TypeInfo&		typeInfo()			const	;
    std::istream&		restore(std::istream& in)		;
    std::ostream&		save(std::ostream& out)		const	;
    __PORT std::istream&	restoreData(std::istream& in)		;
    __PORT std::ostream&	saveData(std::ostream& out)	const	;
    
  private:
    __PORT virtual u_int	_width()			const	;
    __PORT virtual u_int	_height()			const	;
    __PORT virtual Type		_defaultType()			const	;
    __PORT virtual void		_resize(u_int h, u_int w,
					const TypeInfo& typeInfo)	;

    TypeInfo			_typeInfo;
    Array<BGRA>			_colormap;
};

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
#endif	/* !__TUImagePP_h */
