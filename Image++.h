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
 *  $Id: Image++.h,v 1.46 2009-07-31 07:04:44 ueshiba Exp $
 */
#ifndef	__TUImagePP_h
#define	__TUImagePP_h

#include <string.h>
#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  struct RGB, BGR, RGBA & ABGR						*
************************************************************************/
struct BGR;
struct RGBA;
struct ABGR;
struct YUV444;

//! Red, Green, Blue（各8bit）の順で並んだカラー画素
struct RGB
{
    RGB()					:r(0),  g(0),  b(0)	{}
    RGB(u_char rr, u_char gg, u_char bb)	:r(rr), g(gg), b(bb)	{}
    RGB(const BGR& p)							;
    RGB(const RGBA& p)							;
    RGB(const ABGR& p)							;
    RGB(const YUV444& p)						;
    template <class T>
    RGB(const T& p)	:r(u_char(p)), g(u_char(p)), b(u_char(p))	{}

		operator u_char()	const	{return u_char(double(*this));}
		operator short()	const	{return short(double(*this));}
		operator int()		const	{return int(double(*this));}
		operator float()	const	{return float(double(*this));}
		operator double()	const	{return 0.3*r+0.59*g+0.11*b+0.5;}
    
    RGB&	operator +=(const RGB& p)	{r += p.r; g += p.g; b += p.b;
						 return *this;}
    RGB&	operator -=(const RGB& p)	{r -= p.r; g -= p.g; b -= p.b;
						 return *this;}
    bool	operator ==(const RGB& p) const	{return (r == p.r &&
							 g == p.g &&
							 b == p.b);}
    bool	operator !=(const RGB& p) const	{return !(*this == p);}
    
    u_char	r, g, b;
};

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

//! Blue, Green, Red（各8bit）の順で並んだカラー画素
struct BGR
{
    BGR()					:b(0),   g(0),   r(0)	{}
    BGR(u_char rr, u_char gg, u_char bb)	:b(bb),  g(gg),  r(rr)	{}
    BGR(const RGB& p)				:b(p.b), g(p.g), r(p.r)	{}
    BGR(const RGBA& p)							;
    BGR(const ABGR& p)							;
    BGR(const YUV444& p)						;
    template <class T>
    BGR(const T& c)	:b(u_char(c)), g(u_char(c)), r(u_char(c))	{}

		operator u_char()	const	{return u_char(double(*this));}
		operator short()	const	{return short(double(*this));}
		operator int()		const	{return int(double(*this));}
		operator float()	const	{return float(double(*this));}
		operator double()	const	{return 0.3*r+0.59*g+0.11*b+0.5;}

    BGR&	operator +=(const BGR& p)	{r += p.r; g += p.g; b += p.b;
						 return *this;}
    BGR&	operator -=(const BGR& p)	{r -= p.r; g -= p.g; b -= p.b;
						 return *this;}
    bool	operator ==(const BGR& p) const	{return (r == p.r &&
							 g == p.g &&
							 b == p.b);}
    bool	operator !=(const BGR& p) const	{return !(*this != p);}
    
    u_char	b, g, r;
};

inline
RGB::RGB(const BGR& p)	:r(p.r), g(p.g), b(p.b)	{}

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

//! カラー画素のalphaチャンネル
struct Alpha
{
    Alpha(u_char aa=255)	:a(aa)			{}

    int		operator ==(const Alpha& p) const	{return a == p.a;}
    int		operator !=(const Alpha& p) const	{return !(*this == p);}
    
    u_char	a;
};

//! Red, Green, Blue, Alpha（各8bit）の順で並んだカラー画素
struct RGBA : public RGB, public Alpha
{
    RGBA()		:RGB(),        Alpha()		{}
    RGBA(u_char r, u_char g, u_char b, u_char a=255)
			:RGB(r, g, b), Alpha(a)		{}
    template <class T>
    RGBA(const T& p)	:RGB(p),       Alpha()		{}

    bool	operator ==(const RGBA& p) const
		{return (Alpha::operator ==(p) && RGB::operator ==(p));}
    bool	operator !=(const RGBA& p)	const	{return !(*this != p);}
};

//! Alpha, Blue, Green, Red（各8bit）の順で並んだカラー画素
struct ABGR : public Alpha, public BGR
{
    ABGR()		:Alpha(),  BGR()		{}
    ABGR(u_char r, u_char g, u_char b, u_char a=255)
			:Alpha(a), BGR(r, g, b)  	{}
    template <class T>
    ABGR(const T& p)	:Alpha(),  BGR(p)		{}

    bool	operator ==(const ABGR& p) const
		{return (Alpha::operator ==(p) && BGR::operator ==(p));}
    bool	operator !=(const ABGR& p) const	{return !(*this != p);}
};

inline
RGB::RGB(const RGBA& p)	:r(p.r), g(p.g), b(p.b)	{}

inline
RGB::RGB(const ABGR& p)	:r(p.r), g(p.g), b(p.b)	{}

inline
BGR::BGR(const RGBA& p)	:r(p.r), g(p.g), b(p.b)	{}

inline
BGR::BGR(const ABGR& p)	:r(p.r), g(p.g), b(p.b)	{}

/************************************************************************
*  struct YUV444, YUV422, YUV411					*
************************************************************************/
//! Y, U, V（各8bit）の順で並んだカラー画素
struct YUV444
{
    YUV444(u_char yy=0, u_char uu=128, u_char vv=128)
			:u(uu), y(yy), v(vv)		{}
    template <class T> 
    YUV444(const T& p)	:u(128), y(u_char(p)), v(128)	{}

		operator u_char()		const	{return u_char(y);}
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
    template <class T>
    YUV422(const T& p)		:x(128), y(u_char(p))	{}

		operator u_char()		const	{return u_char(y);}
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

//! [U, Y0, Y1], [V, Y2, Y3]（各8bit）の順で並んだカラー画素(12bits/pixel)
struct YUV411
{
    YUV411(u_char yy0=0, u_char yy1=0, u_char xx=128)
			:x(xx), y0(yy0), y1(yy1)		{}
    template <class T>
    YUV411(const T& p)	:x(128), y0(u_char(p)), y1(u_char(p))	{}

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

/************************************************************************
*  function fromYUV<T>()						*
************************************************************************/
//! カラーのY, U, V値を与えて他のカラー表現に変換するクラス
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

inline
RGB::RGB(const YUV444& p)
{
    *this = fromYUV<RGB>(p.y, p.u, p.v);
}

inline
BGR::BGR(const YUV444& p)
{
    *this = fromYUV<BGR>(p.y, p.u, p.v);
}

/************************************************************************
*  class ImageBase:	basic image class				*
************************************************************************/
//! 画素の2次元配列として定義されたあらゆる画像の基底となるクラス
class ImageBase
{
  public:
  //! 外部記憶に読み書きする際の画素のタイプ
    enum Type
    {
	DEFAULT = 0,	//!< same as internal type
	U_CHAR	= 5,	//!< unsigned mono  8bit/pixel
	RGB_24	= 6,	//!< RGB	   24bit/pixel	
	SHORT,		//!< signed mono   16bit/pixel
	INT,		//!< signed mono   32bit/pixel	
	FLOAT,		//!< float mono	   32bit/pixel 
	DOUBLE,		//!< double mono   64bit/pixel
	YUV_444,	//!< YUV444	   24bit/pixel
	YUV_422,	//!< YUV422	   16bit/pixel
	YUV_411		//!< YUV411	   12bit/pixel
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

    static u_int	type2depth(Type type)			;
    
  public:
    Type		restoreHeader(std::istream& in)		;
    Type		saveHeader(std::ostream& out,
				   Type type=DEFAULT)	const	;

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
    virtual u_int	_width()			const	= 0;
    virtual u_int	_height()			const	= 0;
    virtual Type	_defaultType()			const	= 0;
    virtual void	_resize(u_int h, u_int w, Type type)	= 0;

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
  public:
  //! 指定した画素数のスキャンラインを生成する．
  /*!
    \param d	画素数
  */
    explicit ImageLine(u_int d=0)
        :Array<T>(d), _lmost(0), _rmost(d)		{*this = 0;}

  //! 外部の領域と画素数を指定してスキャンラインを生成する．
  /*!
    \param p	外部領域へのポインタ
    \param d	画素数
  */
    ImageLine(T* p, u_int d)
        :Array<T>(p, d), _lmost(0), _rmost(d)		{}

  //! 全ての画素に同一の値を代入する．
  /*!
    \param c	代入する画素値
    \return	このスキャンライン
  */
    ImageLine&		operator =(const T& c)
			{
			    Array<T>::operator =(c);
			    return *this;
			}

    using		Array<T>::dim;
    template <class S>
    T			at(S uf)		const	;
    const YUV422*	fill(const YUV422* src)		;
    const YUV411*	fill(const YUV411* src)		;
    const T*		fill(const T* src)		;
    template <class S>
    const S*		fill(const S* src)		;

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
    const T*	in = &(*this)[u];
    const S	du = uf - u;
    return (du ? *in + du*(*(in + 1) - *in) : *in);
}

//! ポインタで指定された位置からスキャンラインの画素数分の画素を読み込む．
/*!
  \param src	読み込み元の先頭を指すポインタ
  \return	最後に読み込まれた画素の次の画素へのポインタ
*/
template <class T> const YUV422*
ImageLine<T>::fill(const YUV422* src)
{
    register T* dst = *this;
    for (register u_int u = 0; u < dim(); u += 2)
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
    register T*  dst = *this;
    for (register u_int u = 0; u < dim(); u += 4)
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
    T* dst = *this;
    for (u_int n = dim() + 1; --n; )
	*dst++ = T(*src++);
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
    memcpy((T*)*this, src, dim() * sizeof(T));
    return src + dim();
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
    return Array<T>::resize(d);
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
    ImageLine&		operator =(YUV422 c)
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

template <class S> const S*
ImageLine<YUV422>::fill(const S* src)
{
    YUV422* dst = *this;
    for (u_int n = dim() + 1; --n; )
	*dst++ = YUV422(*src++);
    return src;
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
    ImageLine&		operator =(YUV411 c)
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

template <class S> const S*
ImageLine<YUV411>::fill(const S* src)
{
    YUV411* dst = *this;
    for (u_int n = dim() + 1; --n; )
	*dst++ = YUV411(*src++);
    return src;
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
//! T型の画素を持つ画像を表すクラス
/*!
  \param T	画素の型
  \param B	バッファの型
*/
template <class T, class B=Buf<T> >
class Image : public Array2<ImageLine<T>, B>, public ImageBase
{
  public:
  //! 幅と高さを指定して画像を生成する．
  /*!
    \param w	画像の幅
    \param h	画像の高さ
  */
    explicit Image(u_int w=0, u_int h=0)
	:Array2<ImageLine<T>, B>(h, w), ImageBase()		{*this = 0;}

  //! 外部の領域と幅および高さを指定して画像を生成する．
  /*!
    \param p	外部領域へのポインタ
    \param w	画像の幅
    \param h	画像の高さ
  */
    Image(T* p, u_int w, u_int h)			
	:Array2<ImageLine<T>, B>(p, h, w), ImageBase()		{}

  //! 指定された画像の部分画像を生成する．
  /*!
    \param i	元の画像
    \param u	部分画像の左上端の横座標
    \param v	部分画像の左上端の縦座標
    \param w	部分画像の幅
    \param h	部分画像の高さ
  */
    template <class B2>
    Image(const Image<T, B2>& i, u_int u, u_int v, u_int w, u_int h)
	:Array2<ImageLine<T>, B>(i, v, u, h, w), ImageBase(i)	{}

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
    
    u_int	width()			const	{return
						 Array2<ImageLine<T> >::ncol();}
    u_int	height()		const	{return
						 Array2<ImageLine<T> >::nrow();}
    
  //! 全ての画素に同一の値を代入する．
  /*!
    \param c	代入する画素値
    \return	この画像
  */
    Image&	operator = (const T& c)		{Array2<ImageLine<T> >::
						 operator =(c); return *this;}
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out,
			     Type type=DEFAULT)			const	;
    std::istream&	restoreData(std::istream& in,
				    Type type=DEFAULT)			;
    std::ostream&	saveData(std::ostream& out,
				 Type type=DEFAULT)		const	;
    void		resize(u_int h, u_int w)			;
    void		resize(T* p, u_int h, u_int w)			;

  private:
    template <class S>
    std::istream&	restoreRows(std::istream& in)			;
    template <class D>
    std::ostream&	saveRows(std::ostream& out)		const	;
    Type		defaultType()				const	;
    
    virtual u_int	_width()				const	;
    virtual u_int	_height()				const	;
    virtual Type	_defaultType()				const	;
    virtual void	_resize(u_int h, u_int w, Type)			;
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
    return Image<T>(*this, u, v, w, h);
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
    const int	v  = floor(p[1]);
    const T	out0 = (*this)[v  ].at(p[0]);
    const S	dv = p[1] - v;
    return (dv ? out0 + dv*((*this)[v+1].at(p[0]) - out0) : out0);
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
  \param type	画素タイプ．ただし，#DEFAULTを指定した場合は，
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
  \param in	入力ストリーム
  \param type	ストリーム中のデータの画素タイプ(読み込み先の画像の画素タイプではない)
  \return	inで指定した入力ストリーム
*/
template <class T, class B> std::istream&
Image<T, B>::restoreData(std::istream& in, Type type)
{
    if (type == DEFAULT)
	type = defaultType();
    
    switch (type)
    {
      case U_CHAR:
	return restoreRows<u_char>(in);
      case SHORT:
	return restoreRows<short>(in);
      case INT:
	return restoreRows<int>(in);
      case FLOAT:
	return restoreRows<float>(in);
      case DOUBLE:
	return restoreRows<double>(in);
      case RGB_24:
	return restoreRows<RGB>(in);
      case YUV_444:
	return restoreRows<YUV444>(in);
      case YUV_422:
	return restoreRows<YUV422>(in);
      case YUV_411:
	return restoreRows<YUV411>(in);
      default:
	throw std::runtime_error("Image<T, B>::restoreData(): unknown pixel type!!");
	break;
    }
    return in;
}

//! 指定した画素タイプで出力ストリームに画像の画素データを書き出す．
/*!
  \param out	出力ストリーム
  \param type	画素タイプ．ただし，#DEFAULTを指定した場合は，
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
	return saveRows<u_char>(out);
      case SHORT:
	return saveRows<short>(out);
      case INT:
	return saveRows<int>(out);
      case FLOAT:
	return saveRows<float>(out);
      case DOUBLE:
	return saveRows<double>(out);
      case RGB_24:
	return saveRows<RGB>(out);
      case YUV_444:
	return saveRows<YUV444>(out);
      case YUV_422:
	return saveRows<YUV422>(out);
      case YUV_411:
	return saveRows<YUV411>(out);
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
    Array2<ImageLine<T>, B>::resize(h, w);
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
    Array2<ImageLine<T>, B>::resize(p, h, w);
}
 
template <class T, class B> template <class S> std::istream&
Image<T, B>::restoreRows(std::istream& in)
{
    ImageLine<S>	buf(width());
    for (u_int v = 0; v < height(); )
    {
	if (!buf.restore(in))
	    break;
	(*this)[v++].fill((S*)buf);
    }
    return in;
}

template <class T, class B> template <class D> std::ostream&
Image<T, B>::saveRows(std::ostream& out) const
{
    ImageLine<D>	buf(width());
    for (u_int v = 0; v < height(); )
    {
	buf.fill((const T*)(*this)[v++]);
	if (!buf.save(out))
	    break;
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
Image<YUV411, Buf<YUV411> >::defaultType() const
{
    return YUV_411;
}

template <class T, class B> void
Image<T, B>::_resize(u_int h, u_int w, Type)
{
    Image<T, B>::resize(h, w);		// Don't call ImageBase::resize!
}

template <> inline
Image<YUV411, Buf<YUV411> >::Image(u_int w, u_int h)
    :Array2<ImageLine<YUV411>, Buf<YUV411> >(h, w/2), ImageBase()
{
    *this = 0;
}

template <> inline
Image<YUV411, Buf<YUV411> >::Image(YUV411* p, u_int w, u_int h)
    :Array2<ImageLine<YUV411>, Buf<YUV411> >(p, h, w/2), ImageBase()
{
}

template <> template <class B2> inline
Image<YUV411, Buf<YUV411> >::Image(const Image<YUV411, B2>& i,
				   u_int u, u_int v, u_int w, u_int h)
    :Array2<ImageLine<YUV411>, Buf<YUV411> >(i, v, u/2, h, w/2), ImageBase(i)
{
}

template <> inline u_int
Image<YUV411, Buf<YUV411> >::width() const
{
    return 2 * ncol();
}

template <> inline void
Image<YUV411, Buf<YUV411> >::resize(u_int h, u_int w)
{
    Array2<ImageLine<YUV411>, Buf<YUV411> >::resize(h, w/2);
}

template <> inline void
Image<YUV411, Buf<YUV411> >::resize(YUV411* p, u_int h, u_int w)
{
    Array2<ImageLine<YUV411>, Buf<YUV411> >::resize(p, h, w/2);
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
    GenericImage() :_type(U_CHAR)					{}

    Type		type()					const	;
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out)			const	;
    std::istream&	restoreData(std::istream& in)			;
    std::ostream&	saveData(std::ostream& out)		const	;
    
  private:
    virtual u_int	_width()				const	;
    virtual u_int	_height()				const	;
    virtual Type	_defaultType()				const	;
    virtual void	_resize(u_int h, u_int w,
				ImageBase::Type type)			;

    Type		_type;
};

//! 現在保持している画像の画素タイプを返す．
/*!
  \return	画素タイプ
*/
inline ImageBase::Type
GenericImage::type() const
{
    return _type;
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
    saveHeader(out, _type);
    return saveData(out);
}

}
#endif	/* !__TUImagePP_h */
