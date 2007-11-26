/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，使用，第三者へ開示する
 *  等の著作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: Vector++.h,v 1.22 2007-11-26 07:55:48 ueshiba Exp $
 */
#ifndef __TUVectorPP_h
#define __TUVectorPP_h

#include <cmath>
#ifdef WIN32
#  define M_PI	3.14159265358979323846
#endif
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class Rotation							*
************************************************************************/
//! 2次元超平面内での回転を表すクラス
/*!
  具体的には
  \f[
    \TUvec{R}{}(p, q; \theta) \equiv
    \begin{array}{r@{}l}
      & \begin{array}{ccccccccccc}
        & & \makebox[4.0em]{} & p & & & \makebox[3.8em]{} & q & & &
      \end{array} \\
      \begin{array}{l}
        \\ \\ \\ \raisebox{1.5ex}{$p$} \\ \\ \\ \\ \raisebox{1.5ex}{$q$} \\ \\ \\
      \end{array} &
      \TUbeginarray{ccccccccccc}
	1 \\
	& \ddots \\
	& & 1 \\
	& & & \cos\theta & & & & -\sin\theta \\
	& & & & 1 \\
	& & & & & \ddots \\
	& & & & & & 1 \\
	& & & \sin\theta & & & & \cos\theta \\
	& & & & & & & & 1\\
	& & & & & & & & & \ddots \\
	& & & & & & & & & & 1
      \TUendarray
    \end{array}
  \f]
  なる回転行列で表される．
*/
class Rotation
{
  public:
    Rotation(int p, int q, double x, double y)		;
    Rotation(int p, int q, double theta)		;

  //! p軸を返す．
  /*!
    \return	p軸のindex
  */
    int		p()				const	{return _p;}

  //! q軸を返す．
  /*!
    \return	q軸のindex
  */
    int		q()				const	{return _q;}

  //! 回転角生成ベクトルの長さを返す．
  /*!
    \return	回転角生成ベクトル(x, y)に対して\f$\sqrt{x^2 + y^2}\f$
  */
    double	length()			const	{return _l;}

  //! 回転角のcos値を返す．
  /*!
    \return	回転角のcos値
  */
    double	cos()				const	{return _c;}

  //! 回転角のsin値を返す．
  /*!
    \return	回転角のsin値
  */
    double	sin()				const	{return _s;}
    
  private:
    const int	_p, _q;				// rotation axis
    double	_l;				// length of (x, y)
    double	_c, _s;				// cos & sin
};

/************************************************************************
*  class Vector<T>							*
************************************************************************/
template <class T, class B>	class Matrix;

//! T型の要素を持つベクトルを表すクラス
/*!
  \param T	要素の型
  \param B	バッファ
*/
template <class T, class B=Buf<T> >
class Vector : public Array<T, B>
{
  public:
    typedef typename Array<T, B>::value_type		value_type;
    typedef typename Array<T, B>::difference_type	difference_type;
    typedef typename Array<T, B>::reference		reference;
    typedef typename Array<T, B>::const_reference	const_reference;
    typedef typename Array<T, B>::pointer		pointer;
    typedef typename Array<T, B>::const_pointer		const_pointer;
    typedef typename Array<T, B>::iterator		iterator;
    typedef typename Array<T, B>::const_iterator	const_iterator;
    
  public:
    Vector()								;
    explicit Vector(u_int d)						;
    Vector(T* p, u_int d)						;
    template <class B2>
    Vector(const Matrix<T, B2>& m)					;
    template <class B2>
    Vector(const Vector<T, B2>& v, int i, u_int d)			;
    template <class T2, class B2>
    Vector(const Vector<T2, B2>& v)					;
    template <class T2, class B2>
    Vector&		operator =(const Vector<T2, B2>& v)		;

    using		Array<T, B>::begin;
    using		Array<T, B>::end;
    using		Array<T, B>::size;
    using		Array<T, B>::dim;
  //    using		Array<T, B>::operator pointer;
  //    using		Array<T, B>::operator const_pointer;

    const Vector<T>	operator ()(int i, u_int d)		const	;
    Vector<T>		operator ()(int i, u_int d)			;
    Vector&		operator  =(T c)				;
    Vector&		operator *=(double c)				;
    Vector&		operator /=(double c)				;
    template <class T2, class B2>
    Vector&		operator +=(const Vector<T2, B2>& v)		;
    template <class T2, class B2>
    Vector&		operator -=(const Vector<T2, B2>& v)		;
    template <class T2, class B2>
    Vector&		operator ^=(const Vector<T2, B2>& V)		;
    template <class T2, class B2>
    Vector&		operator *=(const Matrix<T2, B2>& m)		;
    Vector		operator  -()				const	;
    T			square()				const	;
    double		length()				const	;
    template <class T2, class B2>
    T			sqdist(const Vector<T2, B2>& v)		const	;
    template <class T2, class B2>
    double		dist(const Vector<T2, B2>& v)		const	;
    Vector&		normalize()					;
    Vector		normal()				const	;
    template <class T2, class B2>
    Vector&		solve(const Matrix<T2, B2>& m)			;
    Matrix<T, Buf<T> >	skew()					const	;
    void		resize(u_int d)					;
    void		resize(T* p, u_int d)				;
};

//! ベクトルを生成し，全要素を0で初期化する．
template <class T, class B>
Vector<T, B>::Vector()
    :Array<T, B>()
{
    *this = 0;
}

//! 指定された次元のベクトルを生成し，全要素を0で初期化する．
/*!
  \param d	ベクトルの次元
*/
template <class T, class B> inline
Vector<T, B>::Vector(u_int d)
    :Array<T, B>(d)
{
    *this = 0;
}

//! 外部記憶領域と次元を指定してベクトルを生成する．
/*!
  \param p	外部記憶領域へのポインタ
  \param d	ベクトルの次元
*/
template <class T, class B> inline
Vector<T, B>::Vector(T* p, u_int d)
    :Array<T, B>(p, d)
{
}

//! 与えられた行列の行を並べて記憶領域を共有するベクトルを生成する．
/*!
  \param m	記憶領域を共有する行列．全行の記憶領域は連続していなければ
		ならない．
*/
template <class T, class B> template <class B2> inline
Vector<T, B>::Vector(const Matrix<T, B2>& m)
    :Array<T, B>(const_cast<T*>((const T*)m), m.nrow()*m.ncol())
{
}

//! 与えられたベクトルと記憶領域を共有する部分ベクトルを生成する．
/*!
  \param v	元のベクトル
  \param i	部分ベクトルの第0要素を指定するindex
  \param d	部分ベクトルの次元
*/
template <class T, class B> template <class B2> inline
Vector<T, B>::Vector(const Vector<T, B2>& v, int i, u_int d)
    :Array<T, B>(v, i, d)
{
}

//! 他のベクトルと同一要素を持つベクトルを作る(コピーコンストラクタの拡張)．
/*!
  \param v	コピー元ベクトル
*/
template <class T, class B> template <class T2, class B2> inline
Vector<T, B>::Vector(const Vector<T2, B2>& v)
    :Array<T, B>(v)
{
}
    
//! 他のベクトルを自分に代入する(代入演算子の拡張)．
/*!
  \param v	コピー元ベクトル
  \return	このベクトル
*/
template <class T, class B> template <class T2, class B2> inline Vector<T, B>&
Vector<T, B>::operator =(const Vector<T2, B2>& v)
{
    Array<T, B>::operator =(v);
    return *this;
}

//! このベクトルと記憶領域を共有した部分ベクトルを生成する．
/*!
    \param i	部分ベクトルの第0要素を指定するindex
    \param d	部分ベクトルの次元
    \return	生成された部分ベクトル
*/
template <class T, class B> inline Vector<T>
Vector<T, B>::operator ()(int i, u_int d)
{
    return Vector<T>(*this, i, d);
}

//! このベクトルと記憶領域を共有した部分ベクトルを生成する．
/*!
    \param i	部分ベクトルの第0要素を指定するindex
    \param d	部分ベクトルの次元
    \return	生成された部分ベクトル
*/
template <class T, class B> inline const Vector<T>
Vector<T, B>::operator ()(int i, u_int d) const
{
    return Vector<T>(*this, i, d);
}

//! このベクトルの全ての要素に同一の数値を代入する．
/*!
  \param c	代入する数値
  \return	このベクトル
*/
template <class T, class B> inline Vector<T, B>&
Vector<T, B>::operator =(T c)
{
    Array<T, B>::operator =(c);
    return *this;
}

//! このベクトルに指定された数値を掛ける．
/*!
  \param c	掛ける数値
  \return	このベクトル，すなわち\f$\TUvec{u}{}\leftarrow c\TUvec{u}{}\f$
*/
template <class T, class B> inline Vector<T, B>&
Vector<T, B>::operator *=(double c)
{
    Array<T, B>::operator *=(c);
    return *this;
}

//! このベクトルを指定された数値で割る．
/*!
  \param c	割る数値
  \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \frac{\TUvec{u}{}}{c}\f$
*/
template <class T, class B> inline Vector<T, B>&
Vector<T, B>::operator /=(double c)
{
    Array<T, B>::operator /=(c);
    return *this;
}

//! このベクトルに他のベクトルを足す．
/*!
  \param v	足すベクトル
  \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \TUvec{u}{} + \TUvec{v}{}\f$
*/
template <class T, class B> template <class T2, class B2> inline Vector<T, B>&
Vector<T, B>::operator +=(const Vector<T2, B2>& v)
{
    Array<T, B>::operator +=(v);
    return *this;
}

//! このベクトルから他のベクトルを引く．
/*!
  \param v	引くベクトル
  \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \TUvec{u}{} - \TUvec{v}{}\f$
*/
template <class T, class B> template <class T2, class B2> inline Vector<T, B>&
Vector<T, B>::operator -=(const Vector<T2, B2>& v)
{
    Array<T, B>::operator -=(v);
    return *this;
}

//! このベクトルと他の3次元ベクトルとのベクトル積をとる．
/*!
    \param v	他のベクトル
    \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \TUvec{u}{}\times\TUvec{v}{}\f$
    \throw std::invalid_argument	このベクトルとvが3次元でない場合に送出
*/
template <class T, class B> template <class T2, class B2> Vector<T, B>&
Vector<T, B>::operator ^=(const Vector<T2, B2>& v)	// outer product
{
    check_dim(v.dim());
    if (dim() != 3)
	throw std::invalid_argument("TU::Vector<T, B>::operator ^=: dimension must be 3");
    Vector<T, FixedSizedBuf<T, 3u> > tmp(*this);
    (*this)[0] = tmp[1] * v[2] - tmp[2] * v[1];
    (*this)[1] = tmp[2] * v[0] - tmp[0] * v[2];
    (*this)[2] = tmp[0] * v[1] - tmp[1] * v[0];
    return *this;
}

//! このベクトルの右から行列を掛ける．
/*!
  \param m	掛ける行列
  \return	このベクトル，すなわち
		\f$\TUtvec{u}{} \leftarrow \TUtvec{u}{}\TUvec{M}{}\f$
*/
template <class T, class B> template <class T2, class B2> inline Vector<T, B>&
Vector<T, B>::operator *=(const Matrix<T2, B2>& m)
{
    return *this = *this * m;
}

//! このベクトルの符号を反転したベクトルを返す．
/*!
  \return	符号を反転したベクトル，すなわち\f$-\TUvec{u}{}\f$
*/
template <class T, class B> inline Vector<T, B>
Vector<T, B>::operator -() const
{
    return Vector(*this) *= -1;
}

//! このベクトルの長さの2乗を返す．
/*!
  \return	ベクトルの長さの2乗，すなわち\f$\TUnorm{\TUvec{u}{}}^2\f$
*/
template <class T, class B> inline T
Vector<T, B>::square() const
{
    return *this * *this;
}

//! このベクトルの長さを返す．
/*!
  \return	ベクトルの長さ，すなわち\f$\TUnorm{\TUvec{u}{}}\f$
*/
template <class T, class B> inline double
Vector<T, B>::length() const
{
    return sqrt(square());
}

//! このベクトルと他のベクトルの差の長さの2乗を返す．
/*!
  \param v	比較対象となるベクトル
  \return	ベクトル間の差の2乗，すなわち
		\f$\TUnorm{\TUvec{u}{} - \TUvec{v}{}}^2\f$
*/
template <class T, class B> template <class T2, class B2> inline T
Vector<T, B>::sqdist(const Vector<T2, B2>& v) const
{
    return (*this - v).square();
}

//! このベクトルと他のベクトルの差の長さを返す．
/*!
  \param v	比較対象となるベクトル
  \return	ベクトル間の差，すなわち
		\f$\TUnorm{\TUvec{u}{} - \TUvec{v}{}}\f$
*/
template <class T, class B> template <class T2, class B2> inline double
Vector<T, B>::dist(const Vector<T2, B2>& v) const
{
    return sqrt(sqdist(v));
}

//! このベクトルの長さを1に正規化する．
/*!
  \return	このベクトル，すなわち
		\f$
		  \TUvec{u}{}\leftarrow\frac{\TUvec{u}{}}{\TUnorm{\TUvec{u}{}}}
		\f$
*/
template <class T, class B> inline Vector<T, B>&
Vector<T, B>::normalize()
{
    return *this /= length();
}

//! このベクトルの長さを1に正規化したベクトルを返す．
/*!
  \return	長さを正規化したベクトル，すなわち
		\f$\frac{\TUvec{u}{}}{\TUnorm{\TUvec{u}{}}}\f$
*/
template <class T, class B> inline Vector<T, B>
Vector<T, B>::normal() const
{
    return Vector(*this).normalize();
}

//! ベクトルの次元を変更し，全要素を0に初期化する．
/*!
  ただし，他のオブジェクトと記憶領域を共有しているベクトルの次元を
  変更することはできない．
  \param d	新しい次元
*/
template <class T, class B> inline void
Vector<T, B>::resize(u_int d)
{
    Array<T, B>::resize(d);
    *this = 0;
}

//! ベクトルが内部で使用する記憶領域を指定したものに変更する．
/*!
  \param p	新しい記憶領域へのポインタ
  \param siz	新しい次元
*/
template <class T, class B> inline void
Vector<T, B>::resize(T* p, u_int d)
{
    Array<T, B>::resize(p, d);
}

/************************************************************************
*  class Matrix<T, B>							*
************************************************************************/
//! T型の要素を持つ行列を表すクラス
/*!
  各行がT型の要素を持つベクトル#TU::Vector<T>になっている．
  \param T	要素の型
  \param B	バッファ
*/
template <class T, class B=Buf<T> >
class Matrix : public Array2<Vector<T>, B>
{
  public:
    typedef typename Array2<Vector<T>, B>::value_type		value_type;
    typedef typename Array2<Vector<T>, B>::difference_type	difference_type;
    typedef typename Array2<Vector<T>, B>::reference		reference;
    typedef typename Array2<Vector<T>, B>::const_reference	const_reference;
    typedef typename Array2<Vector<T>, B>::pointer		pointer;
    typedef typename Array2<Vector<T>, B>::const_pointer	const_pointer;
    typedef typename Array2<Vector<T>, B>::iterator		iterator;
    typedef typename Array2<Vector<T>, B>::const_iterator	const_iterator;
    
  public:
    explicit Matrix(u_int r=0, u_int c=0)				;
    Matrix(T* p, u_int r, u_int c)					;
    template <class B2>
    Matrix(const Matrix<T, B2>& m, int i, int j, u_int r, u_int c)	;
    template <class T2, class B2>
    Matrix(const Matrix<T2, B2>& m)					;
    template <class T2, class B2>
    Matrix&		operator =(const Matrix<T2, B2>& m)		;

    using		Array2<Vector<T>, B>::begin;
    using		Array2<Vector<T>, B>::end;
    using		Array2<Vector<T>, B>::size;
    using		Array2<Vector<T>, B>::dim;
    using		Array2<Vector<T>, B>::nrow;
    using		Array2<Vector<T>, B>::ncol;
  //    using		Array2<Vector<T>, B>::operator T*;
  //    using		Array2<Vector<T>, B>::operator const T*;
    
    const Matrix<T>	operator ()(int i, int j,
				    u_int r, u_int c)		const	;
    Matrix<T>		operator ()(int i, int j,
				    u_int r, u_int c)			;
    Matrix&		operator  =(T c)				;
    Matrix&		operator *=(double c)				;
    Matrix&		operator /=(double c)				;
    template <class T2, class B2>
    Matrix&		operator +=(const Matrix<T2, B2>& m)		;
    template <class T2, class B2>
    Matrix&		operator -=(const Matrix<T2, B2>& m)		;
    template <class T2, class B2>
    Matrix&		operator *=(const Matrix<T2, B2>& m)		;
    template <class T2, class B2>
    Matrix&		operator ^=(const Vector<T2, B2>& v)		;
    Matrix		operator  -()				const	;
    Matrix&		diag(T c)					;
    Matrix<T>		trns()					const	;
    Matrix		inv()					const	;
    template <class T2, class B2>
    Matrix&		solve(const Matrix<T2, B2>& m)			;
    T			det()					const	;
    T			det(int p, int q)			const	;
    T			trace()					const	;
    Matrix		adj()					const	;
    Matrix<T>		pinv(T cndnum=1.0e5)			const	;
    Matrix<T>		eigen(Vector<T>& eval)			const	;
    Matrix<T>		geigen(const Matrix<T>& B,
			       Vector<T>& eval)			const	;
    Matrix		cholesky()				const	;
    Matrix&		normalize()					;
    Matrix&		rotate_from_left(const Rotation& r)		;
    Matrix&		rotate_from_right(const Rotation& r)		;
    T			square()				const	;
    double		length()				const	;
    Matrix&		symmetrize()					;
    Matrix&		antisymmetrize()				;
    void		rot2angle(T& theta_x,
				  T& theta_y, T& theta_z)	const	;
    Vector<T, FixedSizedBuf<T, 3> >
			rot2axis(T& c, T& s)			const	;
    Vector<T, FixedSizedBuf<T, 3> >
			rot2axis()				const	;

    static Matrix	I(u_int d)					;
    template <class T2, class B2>
    static Matrix<T>	Rt(const Vector<T2, B2>& n, T c, T s)		;
    template <class T2, class B2>
    static Matrix<T>	Rt(const Vector<T2, B2>& axis)			;

    void		resize(u_int r, u_int c)			;
    void		resize(T* p, u_int r, u_int c)			;
};

//! 指定されたサイズの行列を生成し，全要素を0で初期化する．
/*!
  \param r	行列の行数
  \param c	行列の列数
*/
template <class T, class B> inline
Matrix<T, B>::Matrix(u_int r, u_int c)
    :Array2<Vector<T>, B>(r, c)
{
    *this = 0;
}

//! 外部記憶領域とサイズを指定して行列を生成する．
/*!
  \param p	外部記憶領域へのポインタ
  \param r	行列の行数
  \param c	行列の列数
*/
template <class T, class B> inline
Matrix<T, B>::Matrix(T* p, u_int r, u_int c)
    :Array2<Vector<T>, B>(p, r, c)
{
}

//! 与えられた行列と記憶領域を共有する部分行列を生成する．
/*!
  \param m	元の行列
  \param i	部分行列の第0行を指定するindex
  \param j	部分行列の第0列を指定するindex
  \param r	部分行列の行数
  \param c	部分行列の列数
*/
template <class T, class B> template <class B2> inline
Matrix<T, B>::Matrix(const Matrix<T, B2>& m, int i, int j, u_int r, u_int c)
    :Array2<Vector<T>, B>(m, i, j, r, c)
{
}

//! 他の行列と同一要素を持つ行列を作る(コピーコンストラクタの拡張)．
/*!
  \param m	コピー元行列
*/
template <class T, class B> template <class T2, class B2> inline
Matrix<T, B>::Matrix(const Matrix<T2, B2>& m)
    :Array2<Vector<T>, B>(m)
{
}

//! 他の行列を自分に代入する(代入演算子の拡張)．
/*!
  \param m	コピー元行列
  \return	この行列
*/
template <class T, class B> template <class T2, class B2> inline Matrix<T, B>&
Matrix<T, B>::operator =(const Matrix<T2, B2>& m)
{
    Array2<Vector<T>, B>::operator =(m);
    return *this;
}

//! この行列と記憶領域を共有した部分行列を生成する．
/*!
    \param i	部分行列の左上隅要素となる行を指定するindex
    \param j	部分行列の左上隅要素となる列を指定するindex
    \param r	部分行列の行数
    \param c	部分行列の列数
    \return	生成された部分行列
*/
template <class T, class B> inline Matrix<T>
Matrix<T, B>::operator ()(int i, int j, u_int r, u_int c)
{
    return Matrix<T>(*this, i, j, r, c);
}

//! この行列と記憶領域を共有した部分行列を生成する．
/*!
    \param i	部分行列の左上隅要素となる行を指定するindex
    \param j	部分行列の左上隅要素となる列を指定するindex
    \param r	部分行列の行数
    \param c	部分行列の列数
    \return	生成された部分行列
*/
template <class T, class B> inline const Matrix<T>
Matrix<T, B>::operator ()(int i, int j, u_int r, u_int c) const
{
    return Matrix<T>(*this, i, j, r, c);
}

//! この行列の全ての要素に同一の数値を代入する．
/*!
  \param c	代入する数値
  \return	この行列
*/
template <class T, class B> inline Matrix<T, B>&
Matrix<T, B>::operator =(T c)
{
    Array2<Vector<T>, B>::operator =(c);
    return *this;
}

//! この行列に指定された数値を掛ける．
/*!
  \param c	掛ける数値
  \return	この行列，すなわち\f$\TUvec{A}{}\leftarrow c\TUvec{A}{}\f$
*/
template <class T, class B> inline Matrix<T, B>&
Matrix<T, B>::operator *=(double c)
{
    Array2<Vector<T>, B>::operator *=(c);
    return *this;
}

//! この行列を指定された数値で割る．
/*!
  \param c	割る数値
  \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \frac{\TUvec{A}{}}{c}\f$
*/
template <class T, class B> inline Matrix<T, B>&
Matrix<T, B>::operator /=(double c)
{
    Array2<Vector<T>, B>::operator /=(c);
    return *this;
}

//! この行列に他の行列を足す．
/*!
  \param m	足す行列
  \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \TUvec{A}{} + \TUvec{M}{}\f$
*/
template <class T, class B> template <class T2, class B2> inline Matrix<T, B>&
Matrix<T, B>::operator +=(const Matrix<T2, B2>& m)
{
    Array2<Vector<T>, B>::operator +=(m);
    return *this;
}

//! この行列から他の行列を引く．
/*!
  \param m	引く行列
  \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \TUvec{A}{} - \TUvec{M}{}\f$
*/
template <class T, class B> template <class T2, class B2> inline Matrix<T, B>&
Matrix<T, B>::operator -=(const Matrix<T2, B2>& m)
{
    Array2<Vector<T>, B>::operator -=(m);
    return *this;
}

//! この行列に他の行列を掛ける．
/*!
  \param m	掛ける行列
  \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \TUvec{A}{}\TUvec{M}{}\f$
*/
template <class T, class B> template <class T2, class B2> inline Matrix<T, B>&
Matrix<T, B>::operator *=(const Matrix<T2, B2>& m)
{
    return *this = *this * m;
}

//! この?x3行列の各行と3次元ベクトルとのベクトル積をとる．
/*!
  \param v	3次元ベクトル
  \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow(\TUtvec{A}{}\times\TUvec{v}{})^\top\f$
*/
template <class T, class B> template <class T2, class B2> Matrix<T, B>&
Matrix<T, B>::operator ^=(const Vector<T2, B2>& v)
{
    for (int i = 0; i < nrow(); ++i)
	(*this)[i] ^= v;
    return *this;
}

//! この行列の符号を反転した行列を返す．
/*!
  \return	符号を反転した行列，すなわち\f$-\TUvec{A}{}\f$
*/
template <class T, class B> inline Matrix<T, B>
Matrix<T, B>::operator -() const
{
    return Matrix(*this) *= -1;
}

//! この行列の逆行列を返す．
/*!
  \return	逆行列，すなわち\f$\TUinv{A}{}\f$
*/
template <class T, class B> inline Matrix<T, B>
Matrix<T, B>::inv() const
{
    return I(nrow()).solve(*this);
}

//! この行列の2乗ノルムを返す．
/*!
  \return	行列の2乗ノルム，すなわち\f$\TUnorm{\TUvec{A}{}}\f$
*/
template <class T, class B> inline double
Matrix<T, B>::length() const
{
    return sqrt(square());
}

//! 単位正方行列を生成する．
/*!
  \param d	単位正方行列の次元
  \return	単位正方行列
*/
template <class T, class B> inline Matrix<T, B>
Matrix<T, B>::I(u_int d)
{
    return Matrix<T, B>(d, d).diag(1.0);
}

//! 3次元回転行列を生成する．
/*!
  \param n	回転軸を表す3次元単位ベクトル
  \param c	回転角のcos値
  \param s	回転角のsin値
  \return	生成された回転行列，すなわち
		\f[
		  \TUtvec{R}{} \equiv \TUvec{I}{3}\cos\theta
		  + \TUvec{n}{}\TUtvec{n}{}(1 - \cos\theta)
		  - \TUskew{n}{}\sin\theta
		\f]
*/
template <class T, class B> template <class T2, class B2> Matrix<T>
Matrix<T, B>::Rt(const Vector<T2, B2>& n, T c, T s)
{
    Matrix<T>	Qt = n % n;
    Qt *= (1.0 - c);
    Qt[0][0] += c;
    Qt[1][1] += c;
    Qt[2][2] += c;
    Qt[0][1] += n[2] * s;
    Qt[0][2] -= n[1] * s;
    Qt[1][0] -= n[2] * s;
    Qt[1][2] += n[0] * s;
    Qt[2][0] += n[1] * s;
    Qt[2][1] -= n[0] * s;

    return Qt;
}

//! 3次元回転行列を生成する．
/*!
  \param axis	回転角と回転軸を表す3次元ベクトル
  \return	生成された回転行列，すなわち
		\f[
		  \TUtvec{R}{} \equiv \TUvec{I}{3}\cos\theta
		  + \TUvec{n}{}\TUtvec{n}{}(1 - \cos\theta)
		  - \TUskew{n}{}\sin\theta,{\hskip 1em}\mbox{where}{\hskip 0.5em}
		  \theta = \TUnorm{\TUvec{a}{}},~
		  \TUvec{n}{} = \frac{\TUvec{a}{}}{\TUnorm{\TUvec{a}{}}}
		\f]
*/
template <class T, class B> template <class T2, class B2> Matrix<T>
Matrix<T, B>::Rt(const Vector<T2, B2>& axis)
{
    const T	theta = axis.length();
    if (theta + 1.0 == 1.0)		// theta << 1 ?
	return I(3);
    else
    {
	T	c = cos(theta), s = sin(theta);
	return Rt(axis / theta, c, s);
    }
}

//! 行列のサイズを変更し，0に初期化する．
/*!
  \param r	新しい行数
  \param c	新しい列数
*/
template <class T, class B> inline void
Matrix<T, B>::resize(u_int r, u_int c)
{
    Array2<Vector<T>, B>::resize(r, c);
    *this = 0;
}

//! 行列の内部記憶領域とサイズを変更する．
/*!
  \param p	新しい内部記憶領域へのポインタ
  \param r	新しい行数
  \param c	新しい列数
*/
template <class T, class B> inline void
Matrix<T, B>::resize(T* p, u_int r, u_int c)
{
    Array2<Vector<T>, B>::resize(p, r, c);
}

/************************************************************************
*  numerical operators							*
************************************************************************/
//! 2つのベクトルの足し算
/*!
  \param v	第1引数
  \param w	第2引数
  \return	結果を格納したベクトル，すなわち\f$\TUvec{v}{}+\TUvec{w}{}\f$
*/
template <class T1, class B1, class T2, class B2> inline Vector<T1, B1>
operator +(const Vector<T1, B1>& v, const Vector<T2, B2>& w)
{
    return Vector<T1, B1>(v) += w;
}

//! 2つのベクトルの引き算
/*!
  \param v	第1引数
  \param w	第2引数
  \return	結果を格納したベクトル，すなわち\f$\TUvec{v}{}-\TUvec{w}{}\f$
*/
template <class T1, class B1, class T2, class B2> inline Vector<T1, B1>
operator -(const Vector<T1, B1>& v, const Vector<T2, B2>& w)
{
    return Vector<T1, B1>(v) -= w;
}

//! ベクトルに定数を掛ける．
/*!
  \param c	掛ける定数
  \param v	ベクトル
  \return	結果を格納したベクトル，すなわち\f$c\TUvec{v}{}\f$
*/
template <class T, class B> inline Vector<T, B>
operator *(double c, const Vector<T, B>& v)
{
    return Vector<T, B>(v) *= c;
}

//! ベクトルに定数を掛ける．
/*!
  \param v	ベクトル
  \param c	掛ける定数
  \return	結果を格納したベクトル，すなわち\f$c\TUvec{v}{}\f$
*/
template <class T, class B> inline Vector<T, B>
operator *(const Vector<T, B>& v, double c)
{
    return Vector<T, B>(v) *= c;
}

//! ベクトルの各要素を定数で割る．
/*!
  \param v	ベクトル
  \param c	割る定数
  \return	結果を格納したベクトル，すなわち\f$\frac{1}{c}\TUvec{v}{}\f$
*/
template <class T, class B> inline Vector<T, B>
operator /(const Vector<T, B>& v, double c)
{
    return Vector<T, B>(v) /= c;
}

//! 2つの行列の足し算
/*!
  \param m	第1引数
  \param n	第2引数
  \return	結果を格納した行列，すなわち\f$\TUvec{M}{}+\TUvec{N}{}\f$
*/
template <class T1, class B1, class T2, class B2> inline Matrix<T1, B1>
operator +(const Matrix<T1, B1>& m, const Matrix<T2, B2>& n)
{
    return Matrix<T1, B1>(m) += n;
}

//! 2つの行列の引き算
/*!
  \param m	第1引数
  \param n	第2引数
  \return	結果を格納した行列，すなわち\f$\TUvec{M}{}-\TUvec{N}{}\f$
*/
template <class T1, class B1, class T2, class B2> inline Matrix<T1, B1>
operator -(const Matrix<T1, B1>& m, const Matrix<T2, B2>& n)
{
    return Matrix<T1, B1>(m) -= n;
}

//! 行列に定数を掛ける．
/*!
  \param c	掛ける定数
  \param m	行列
  \return	結果を格納した行列，すなわち\f$c\TUvec{M}{}\f$
*/
template <class T, class B> inline Matrix<T, B>
operator *(double c, const Matrix<T, B>& m)
{
    return Matrix<T, B>(m) *= c;
}

//! 行列に定数を掛ける．
/*!
  \param m	行列
  \param c	掛ける定数
  \return	結果を格納した行列，すなわち\f$c\TUvec{M}{}\f$
*/
template <class T, class B> inline Matrix<T, B>
operator *(const Matrix<T, B>& m, double c)
{
    return Matrix<T, B>(m) *= c;
}

//! 行列の各要素を定数で割る．
/*!
  \param m	行列
  \param c	割る定数
  \return	結果を格納した行列，すなわち\f$\frac{1}{c}\TUvec{M}{}\f$
*/
template <class T, class B> inline Matrix<T, B>
operator /(const Matrix<T, B>& m, double c)
{
    return Matrix<T, B>(m) /= c;
}

//! 2つの3次元ベクトルのベクトル積
/*!
  \param v	第1引数
  \param w	第2引数
  \return	ベクトル積，すなわち\f$\TUvec{v}{}\times\TUvec{w}{}\f$
*/
template <class T1, class B1, class T2, class B2> inline Vector<T1, B1>
operator ^(const Vector<T1, B1>& v, const Vector<T2, B2>& w)
{
    return Vector<T1, B1>(v) ^= w;
}

//! 2つのベクトルの内積
/*!
  \param v	第1引数
  \param w	第2引数
  \return	内積，すなわち\f$\TUtvec{v}{}\TUvec{w}{}\f$
*/
template <class T1, class B1, class T2, class B2> T1
operator *(const Vector<T1, B1>& v, const Vector<T2, B2>& w)
{
    v.check_dim(w.dim());
    T1	val = 0;
    for (int i = 0; i < v.dim(); ++i)
	val += v[i] * w[i];
    return val;
}

//! ベクトルと行列の積
/*!
  \param v	ベクトル
  \param m	行列
  \return	結果のベクトル，すなわち\f$\TUtvec{v}{}\TUvec{M}{}\f$
*/
template <class T1, class B1, class T2, class B2> Vector<T1, B1>
operator *(const Vector<T1, B1>& v, const Matrix<T2, B2>& m)
{
    v.check_dim(m.nrow());
    Vector<T1, B1> val(m.ncol());
    for (int j = 0; j < m.ncol(); ++j)
	for (int i = 0; i < m.nrow(); ++i)
	    val[j] += v[i] * m[i][j];
    return val;
}

//! 2つのベクトルの外積
/*!
  \param v	第1引数
  \param w	第2引数
  \return	結果の行列，すなわち\f$\TUvec{v}{}\TUtvec{w}{}\f$
*/
template <class T1, class B1, class T2, class B2> Matrix<T1>
operator %(const Vector<T1, B1>& v, const Vector<T2, B2>& w)
{
    Matrix<T1>	val(v.dim(), w.dim());
    for (int i = 0; i < v.dim(); ++i)
	for (int j = 0; j < w.dim(); ++j)
	    val[i][j] = v[i] * w[j];
    return val;
}

//! 3次元ベクトルと3x?行列の各列とのベクトル積
/*!
  \param v			3次元ベクトル
  \param m			3x?行列
  \return			結果の3x?行列，すなわち
				\f$\TUvec{v}{}\times\TUvec{M}{}\f$
  \throw std::invalid_argument	vが3次元ベクトルでないかmが3x?行列でない場合に
				送出
*/
template <class T, class B, class T2> Matrix<T, B>
operator ^(const Vector<T, B>& v, const Matrix<T2>& m)
{
    v.check_dim(m.nrow());
    if (v.dim() != 3)
	throw std::invalid_argument("operator ^(const Vecotr<T>&, const Matrix<T, B>&): dimension of vector must be 3!!");
    Matrix<T, B>	val(m.nrow(), m.ncol());
    for (int j = 0; j < val.ncol(); ++j)
    {
	val[0][j] = v[1] * m[2][j] - v[2] * m[1][j];
	val[1][j] = v[2] * m[0][j] - v[0] * m[2][j];
	val[2][j] = v[0] * m[1][j] - v[1] * m[0][j];
    }
    return val;
}

//! 2つの行列の積
/*!
  \param m	第1引数
  \param n	第2引数
  \return	結果の行列，すなわち\f$\TUvec{M}{}\TUvec{N}{}\f$
*/
template <class T1, class B1, class T2, class B2> Matrix<T1>
operator *(const Matrix<T1, B1>& m, const Matrix<T2, B2>& n)
{
    n.check_dim(m.ncol());
    Matrix<T1>	val(m.nrow(), n.ncol());
    for (int i = 0; i < m.nrow(); ++i)
	for (int j = 0; j < n.ncol(); ++j)
	    for (int k = 0; k < m.ncol(); ++k)
		val[i][j] += m[i][k] * n[k][j];
    return val;
}

//! 行列とベクトルの積
/*!
  \param m	行列
  \param v	ベクトル
  \return	結果のベクトル，すなわち\f$\TUvec{M}{}\TUvec{v}{}\f$
*/
template <class T1, class B1, class T2, class B2> Vector<T1>
operator *(const Matrix<T1, B1>& m, const Vector<T2, B2>& v)
{
    Vector<T1>	val(m.nrow());
    for (int i = 0; i < m.nrow(); ++i)
	val[i] = m[i] * v;
    return val;
}

//! ?x3行列の各行と3次元ベクトルのベクトル積
/*!
  \param m	?x3行列
  \param v	3次元ベクトル
  \return	結果の行列，すなわち\f$(\TUtvec{M}{}\times\TUvec{v}{})^\top\f$
*/
template <class T1, class B1, class T2, class B2> inline Matrix<T1, B1>
operator ^(const Matrix<T1, B1>& m, const Vector<T2, B2>& v)
{
    return Matrix<T1, B1>(m) ^= v;
}

/************************************************************************
*  class LUDecomposition<T>						*
************************************************************************/
//! 正方行列のLU分解を表すクラス
template <class T>
class LUDecomposition : private Array2<Vector<T> >
{
  public:
    template <class T2, class B2>
    LUDecomposition(const Matrix<T2, B2>&)		;

    template <class T2, class B2>
    void	substitute(Vector<T2, B2>&)	const	;

  //! もとの正方行列の行列式を返す．
  /*!
    \return	もとの正方行列の行列式
  */
    T		det()				const	{return _det;}
    
  private:
    using	Array2<Vector<T> >::nrow;
    using	Array2<Vector<T> >::ncol;
    
    Array<int>	_index;
    T		_det;
};

//! 与えられた正方行列のLU分解を生成する．
/*!
 \param m			LU分解する正方行列
 \throw std::invalid_argument	mが正方行列でない場合に送出
*/
template <class T> template <class T2, class B2>
LUDecomposition<T>::LUDecomposition(const Matrix<T2, B2>& m)
    :Array2<Vector<T> >(m), _index(ncol()), _det(1.0)
{
    using namespace	std;
    
    if (nrow() != ncol())
        throw invalid_argument("TU::LUDecomposition<T>::LUDecomposition: not square matrix!!");

    for (int j = 0; j < ncol(); ++j)	// initialize column index
	_index[j] = j;			// for explicit pivotting

    Vector<T>	scale(ncol());
    for (int j = 0; j < ncol(); ++j)	// find maximum abs. value in each col.
    {					// for implicit pivotting
	T max = 0.0;

	for (int i = 0; i < nrow(); ++i)
	{
	    const T tmp = fabs((*this)[i][j]);
	    if (tmp > max)
		max = tmp;
	}
	scale[j] = (max != 0.0 ? 1.0 / max : 1.0);
    }

    for (int i = 0; i < nrow(); ++i)
    {
	for (int j = 0; j < i; ++j)		// left part (j < i)
	{
	    T& sum = (*this)[i][j];
	    for (int k = 0; k < j; ++k)
		sum -= (*this)[i][k] * (*this)[k][j];
	}

	int	jmax;
	T	max = 0.0;
	for (int j = i; j < ncol(); ++j)  // diagonal and right part (i <= j)
	{
	    T& sum = (*this)[i][j];
	    for (int k = 0; k < i; ++k)
		sum -= (*this)[i][k] * (*this)[k][j];
	    const T tmp = fabs(sum) * scale[j];
	    if (tmp >= max)
	    {
		max  = tmp;
		jmax = j;
	    }
	}
	if (jmax != i)			// pivotting required ?
	{
	    for (int k = 0; k < nrow(); ++k)	// swap i-th and jmax-th column
		swap((*this)[k][i], (*this)[k][jmax]);
	    swap(_index[i], _index[jmax]);	// swap column index
	    swap(scale[i], scale[jmax]);	// swap colum-wise scale factor
	    _det = -_det;
	}

	_det *= (*this)[i][i];

	if ((*this)[i][i] == 0.0)	// singular matrix ?
	    break;

	for (int j = i + 1; j < nrow(); ++j)
	    (*this)[i][j] /= (*this)[i][i];
    }
}

//! もとの正方行列を係数行列とした連立1次方程式を解く．
/*!
  \param b			もとの正方行列\f$\TUvec{M}{}\f$と同じ次
				元を持つベクトル．\f$\TUtvec{b}{} =
				\TUtvec{x}{}\TUvec{M}{}\f$の解に変換さ
				れる．
  \throw std::invalid_argument	ベクトルbの次元がもとの正方行列の次元に一致
				しない場合に送出
  \throw std::runtime_error	もとの正方行列が正則でない場合に送出
*/
template <class T> template <class T2, class B2> void
LUDecomposition<T>::substitute(Vector<T2, B2>& b) const
{
    if (b.dim() != ncol())
	throw std::invalid_argument("TU::LUDecomposition<T>::substitute: Dimension of given vector is not equal to mine!!");
    
    Vector<T2, B2>	tmp(b);
    for (int j = 0; j < b.dim(); ++j)
	b[j] = tmp[_index[j]];

    for (int j = 0; j < b.dim(); ++j)		// forward substitution
	for (int i = 0; i < j; ++i)
	    b[j] -= b[i] * (*this)[i][j];
    for (int j = b.dim(); --j >= 0; )		// backward substitution
    {
	for (int i = b.dim(); --i > j; )
	    b[j] -= b[i] * (*this)[i][j];
	if ((*this)[j][j] == 0.0)		// singular matrix ?
	    throw std::runtime_error("TU::LUDecomposition<T>::substitute: singular matrix !!");
	b[j] /= (*this)[j][j];
    }
}

//! 連立1次方程式を解く．
/*!
  \param m	正則な正方行列
  \return	\f$\TUtvec{u}{} = \TUtvec{x}{}\TUvec{M}{}\f$
		の解を納めたこのベクトル，すなわち
		\f$\TUtvec{u}{} \leftarrow \TUtvec{u}{}\TUinv{M}{}\f$
*/
template <class T, class B> template <class T2, class B2> inline Vector<T, B>&
Vector<T, B>::solve(const Matrix<T2, B2>& m)
{
    LUDecomposition<T2>(m).substitute(*this);
    return *this;
}

//! 連立1次方程式を解く．
/*!
  \param m	正則な正方行列
  \return	\f$\TUvec{A}{} = \TUvec{X}{}\TUvec{M}{}\f$
		の解を納めたこの行列，すなわち
		\f$\TUvec{A}{} \leftarrow \TUvec{A}{}\TUinv{M}{}\f$
*/
template <class T, class B> template <class T2, class B2> Matrix<T, B>&
Matrix<T, B>::solve(const Matrix<T2, B2>& m)
{
    LUDecomposition<T2>	lu(m);
    
    for (int i = 0; i < nrow(); ++i)
	lu.substitute((*this)[i]);
    return *this;
}

//! この行列の行列式を返す．
/*!
  \return	行列式，すなわち\f$\det\TUvec{A}{}\f$
*/
template <class T, class B> inline T
Matrix<T, B>::det() const
{
    return LUDecomposition<T>(*this).det();
}

/************************************************************************
*  class Householder<T>							*
************************************************************************/
template <class T>	class QRDecomposition;
template <class T>	class TriDiagonal;
template <class T>	class BiDiagonal;

//! Householder変換を表すクラス
template <class T>
class Householder : public Matrix<T>
{
  private:
    Householder(u_int dd, u_int d)
	:Matrix<T>(dd, dd), _d(d), _sigma(Matrix<T>::nrow())	{}
    template <class T2, class B2>
    Householder(const Matrix<T2, B2>& a, u_int d)		;

    using		Matrix<T>::dim;
    
    void		apply_from_left(Matrix<T>&, int)	;
    void		apply_from_right(Matrix<T>&, int)	;
    void		apply_from_both(Matrix<T>&, int)	;
    void		make_transformation()			;
    const Vector<T>&	sigma()				const	{return _sigma;}
    Vector<T>&		sigma()					{return _sigma;}
    bool		sigma_is_zero(int, T)		const	;

  private:
    const u_int		_d;		// deviation from diagonal element
    Vector<T>		_sigma;

    friend class	QRDecomposition<T>;
    friend class	TriDiagonal<T>;
    friend class	BiDiagonal<T>;
};

template <class T> template <class T2, class B2>
Householder<T>::Householder(const Matrix<T2, B2>& a, u_int d)
    :Matrix<T>(a), _d(d), _sigma(dim())
{
    if (a.nrow() != a.ncol())
	throw std::invalid_argument("TU::Householder<T>::Householder: Given matrix must be square !!");
}

/************************************************************************
*  class QRDecomposition<T>						*
************************************************************************/
//! 一般行列のQR分解を表すクラス
/*!
  与えられた行列\f$\TUvec{A}{} \in \TUspace{R}{m\times n}\f$に対して
  \f$\TUvec{A}{} = \TUtvec{R}{}\TUtvec{Q}{}\f$なる下半三角行列
  \f$\TUtvec{R}{} \in \TUspace{R}{m\times n}\f$と回転行列
  \f$\TUtvec{Q}{} \in \TUspace{R}{n\times n}\f$を求める
  （\f$\TUvec{A}{}\f$の各行を\f$\TUtvec{Q}{}\f$の行の線型結合で表現す
  る）．
 */
template <class T>
class QRDecomposition : private Matrix<T>
{
  public:
    template <class T2, class B2>
    QRDecomposition(const Matrix<T2, B2>&)		;

  //! QR分解の下半三角行列を返す．
  /*!
    \return	下半三角行列\f$\TUtvec{R}{}\f$
  */
    const Matrix<T>&	Rt()			const	{return *this;}

  //! QR分解の回転行列を返す．
  /*!
    \return	回転行列\f$\TUtvec{Q}{}\f$
  */
    const Matrix<T>&	Qt()			const	{return _Qt;}
    
  private:
    using		Matrix<T>::nrow;
    using		Matrix<T>::ncol;
    
    Householder<T>	_Qt;			// rotation matrix
};

//! 与えられた一般行列のQR分解を生成する．
/*!
 \param m	QR分解する一般行列
*/
template <class T> template <class T2, class B2>
QRDecomposition<T>::QRDecomposition(const Matrix<T2, B2>& m)
    :Matrix<T>(m), _Qt(m.ncol(), 0)
{
    u_int	n = std::min(nrow(), ncol());
    for (int j = 0; j < n; ++j)
	_Qt.apply_from_right(*this, j);
    _Qt.make_transformation();
    for (int i = 0; i < n; ++i)
    {
	(*this)[i][i] = _Qt.sigma()[i];
	for (int j = i + 1; j < ncol(); ++j)
	    (*this)[i][j] = 0.0;
    }
}

/************************************************************************
*  class TriDiagonal<T>							*
************************************************************************/
//! 対称行列の3重対角化を表すクラス
/*!
  与えられた対称行列\f$\TUvec{A}{} \in \TUspace{R}{d\times d}\f$に対し
  て\f$\TUtvec{U}{}\TUvec{A}{}\TUvec{U}{}\f$が3重対角行列となるような回
  転行列\f$\TUtvec{U}{} \in \TUspace{R}{d\times d}\f$を求める．
 */
template <class T>
class TriDiagonal
{
  public:
    template <class T2, class B2>
    TriDiagonal(const Matrix<T2, B2>&)			;

  //! 3重対角化される対称行列の次元(= 行数 = 列数)を返す．
  /*!
    \return	対称行列の次元
  */
    u_int		dim()			const	{return _Ut.nrow();}

  //! 3重対角化を行う回転行列を返す．
  /*!
    \return	回転行列
  */
    const Matrix<T>&	Ut()			const	{return _Ut;}

  //! 3重対角行列の対角成分を返す．
  /*!
    \return	対角成分
  */
    const Vector<T>&	diagonal()		const	{return _diagonal;}

  //! 3重対角行列の非対角成分を返す．
  /*!
    \return	非対角成分
  */
    const Vector<T>&	off_diagonal()		const	{return _Ut.sigma();}

    void		diagonalize()			;
    
  private:
    enum		{NITER_MAX = 30};

    bool		off_diagonal_is_zero(int)		const	;
    void		initialize_rotation(int, int, T&, T&)	const	;
    
    Householder<T>	_Ut;
    Vector<T>		_diagonal;
    Vector<T>&		_off_diagonal;
};

//! 与えられた対称行列を3重対角化する．
/*!
  \param a			3重対角化する対称行列
  \throw std::invalid_argument	aが正方行列でない場合に送出
*/
template <class T> template <class T2, class B2>
TriDiagonal<T>::TriDiagonal(const Matrix<T2, B2>& a)
    :_Ut(a, 1), _diagonal(_Ut.nrow()), _off_diagonal(_Ut.sigma())
{
    if (_Ut.nrow() != _Ut.ncol())
        throw std::invalid_argument("TU::TriDiagonal<T>::TriDiagonal: not square matrix!!");

    for (int m = 0; m < dim(); ++m)
    {
	_Ut.apply_from_both(_Ut, m);
	_diagonal[m] = _Ut[m][m];
    }

    _Ut.make_transformation();
}

/************************************************************************
*  class BiDiagonal<T>							*
************************************************************************/
//! 一般行列の2重対角化を表すクラス
/*!
  与えられた一般行列\f$\TUvec{A}{} \in \TUspace{R}{m\times n}\f$に対し
  て\f$\TUtvec{V}{}\TUvec{A}{}\TUvec{U}{}\f$が2重対角行列となるような2
  つの回転行列\f$\TUtvec{U}{} \in \TUspace{R}{n\times n}\f$,
  \f$\TUtvec{V}{} \in \TUspace{R}{m\times m}\f$を求める．\f$m \le n\f$
  の場合は下半三角な2重対角行列に，\f$m > n\f$の場合は上半三角な2重対角
  行列になる．
 */
template <class T>
class BiDiagonal
{
  public:
    template <class T2, class B2>
    BiDiagonal(const Matrix<T2, B2>&)		;

  //! 2重対角化される行列の行数を返す．
  /*!
    \return	行列の行数
  */
    u_int		nrow()		const	{return _Vt.nrow();}

  //! 2重対角化される行列の列数を返す．
  /*!
    \return	行列の列数
  */
    u_int		ncol()		const	{return _Ut.nrow();}

  //! 2重対角化を行うために右から掛ける回転行列の転置を返す．
  /*!
    \return	右から掛ける回転行列の転置
  */
    const Matrix<T>&	Ut()		const	{return _Ut;}

  //! 2重対角化を行うために左から掛ける回転行列を返す．
  /*!
    \return	左から掛ける回転行列
  */
    const Matrix<T>&	Vt()		const	{return _Vt;}

  //! 2重対角行列の対角成分を返す．
  /*!
    \return	対角成分
  */
    const Vector<T>&	diagonal()	const	{return _Dt.sigma();}

  //! 2重対角行列の非対角成分を返す．
  /*!
    \return	非対角成分
  */
    const Vector<T>&	off_diagonal()	const	{return _Et.sigma();}

    void		diagonalize()		;

  private:
    enum		{NITER_MAX = 30};
    
    bool		diagonal_is_zero(int)			const	;
    bool		off_diagonal_is_zero(int)		const	;
    void		initialize_rotation(int, int,
					    T&, T&)	const	;

    Householder<T>	_Dt;
    Householder<T>	_Et;
    Vector<T>&		_diagonal;
    Vector<T>&		_off_diagonal;
    T			_anorm;
    const Matrix<T>&	_Ut;
    const Matrix<T>&	_Vt;
};

//! 与えられた一般行列を2重対角化する．
/*!
  \param a	2重対角化する一般行列
*/
template <class T> template <class T2, class B2>
BiDiagonal<T>::BiDiagonal(const Matrix<T2, B2>& a)
    :_Dt((a.nrow() < a.ncol() ? a.ncol() : a.nrow()), 0),
     _Et((a.nrow() < a.ncol() ? a.nrow() : a.ncol()), 1),
     _diagonal(_Dt.sigma()), _off_diagonal(_Et.sigma()), _anorm(0),
     _Ut(a.nrow() < a.ncol() ? _Dt : _Et),
     _Vt(a.nrow() < a.ncol() ? _Et : _Dt)
{
    if (nrow() < ncol())
	for (int i = 0; i < nrow(); ++i)
	    for (int j = 0; j < ncol(); ++j)
		_Dt[i][j] = a[i][j];
    else
	for (int i = 0; i < nrow(); ++i)
	    for (int j = 0; j < ncol(); ++j)
		_Dt[j][i] = a[i][j];

  /* Householder reduction to bi-diagonal (off-diagonal in lower part) form */
    for (int m = 0; m < _Et.dim(); ++m)
    {
	_Dt.apply_from_right(_Dt, m);
	_Et.apply_from_left(_Dt, m);
    }

    _Dt.make_transformation();	// Accumulate right-hand transformation: V
    _Et.make_transformation();	// Accumulate left-hand transformation: U

    for (int m = 0; m < _Et.dim(); ++m)
    {
	T	anorm = fabs(_diagonal[m]) + fabs(_off_diagonal[m]);
	if (anorm > _anorm)
	    _anorm = anorm;
    }
}

/************************************************************************
*  class SVDecomposition<T>						*
************************************************************************/
//! 一般行列の特異値分解を表すクラス
/*!
  与えられた一般行列\f$\TUvec{A}{} \in \TUspace{R}{m\times n}\f$に対し
  て\f$\TUtvec{V}{}\TUvec{A}{}\TUvec{U}{}\f$が対角行列となるような2つの
  回転行列\f$\TUtvec{U}{} \in \TUspace{R}{n\times n}\f$,
  \f$\TUtvec{V}{} \in \TUspace{R}{m\times m}\f$を求める．
 */
template <class T>
class SVDecomposition : private BiDiagonal<T>
{
  public:
  //! 与えられた一般行列の特異値分解を求める．
  /*!
    \param a	特異値分解する一般行列
  */
    template <class T2, class B2>
    SVDecomposition(const Matrix<T2, B2>& a)
	:BiDiagonal<T>(a)		{BiDiagonal<T>::diagonalize();}

    using	BiDiagonal<T>::nrow;
    using	BiDiagonal<T>::ncol;
    using	BiDiagonal<T>::Ut;
    using	BiDiagonal<T>::Vt;
    using	BiDiagonal<T>::diagonal;

  //! 特異値を求める．
  /*!
    \param i	絶対値の大きい順に並んだ特異値の1つを指定するindex
    \return	指定されたindexに対応する特異値
  */
    const T&	operator [](int i)	const	{return diagonal()[i];}
};

/************************************************************************
*  typedefs								*
************************************************************************/
typedef Vector<short,  FixedSizedBuf<short,   2> >	Vector2s;
typedef Vector<int,    FixedSizedBuf<int,     2> >	Vector2i;
typedef Vector<float,  FixedSizedBuf<float,   2> >	Vector2f;
typedef Vector<double, FixedSizedBuf<double,  2> >	Vector2d;
typedef Vector<short,  FixedSizedBuf<short,   3> >	Vector3s;
typedef Vector<int,    FixedSizedBuf<int,     3> >	Vector3i;
typedef Vector<float,  FixedSizedBuf<float,   3> >	Vector3f;
typedef Vector<double, FixedSizedBuf<double,  3> >	Vector3d;
typedef Vector<short,  FixedSizedBuf<short,   4> >	Vector4s;
typedef Vector<int,    FixedSizedBuf<int,     4> >	Vector4i;
typedef Vector<float,  FixedSizedBuf<float,   4> >	Vector4f;
typedef Vector<double, FixedSizedBuf<double,  4> >	Vector4d;
typedef Matrix<float,  FixedSizedBuf<float,   4> >	Matrix22f;
typedef Matrix<double, FixedSizedBuf<double,  4> >	Matrix22d;
typedef Matrix<float,  FixedSizedBuf<float,   9> >	Matrix33f;
typedef Matrix<double, FixedSizedBuf<double,  9> >	Matrix33d;
typedef Matrix<float,  FixedSizedBuf<float,  12> >	Matrix34f;
typedef Matrix<double, FixedSizedBuf<double, 12> >	Matrix34d;
typedef Matrix<float,  FixedSizedBuf<float,  16> >	Matrix44f;
typedef Matrix<double, FixedSizedBuf<double, 16> >	Matrix44d;

/************************************************************************
*  class Minimization1<S>						*
************************************************************************/
static const double	DEFAULT_TOL = 3.0e-8;

template <class S>
class Minimization1
{
  private:
    enum		{DEFAULT_NITER_MAX = 100};

  public:
    Minimization1(S tol = DEFAULT_TOL, int niter_max = DEFAULT_NITER_MAX)
	:_tol(tol), _niter_max(niter_max)				{}
    
    virtual S		operator ()(const S&)			const	= 0;
    S			minimize(S&, S)				const	;

  private:
    const S		_tol;
    const int		_niter_max;
};

/************************************************************************
*  class Minimization<S, T>						*
************************************************************************/
template <class S, class T>
class Minimization
{
  private:
    class LineFunction : public Minimization1<S>
    {
      public:
	LineFunction(const Minimization<S, T>& func,
		     const T& x, const Vector<S>& h,
		     S tol, int niter_max)
	  :Minimization1<S>(tol, niter_max),
	   _func(func), _x(x), _h(h)		{}
    
	S	operator ()(const S& d)	const	{return _func(_func.proceed
							      (_x, d * _h));}
    
      private:
	const Minimization<S, T>&	_func;
	const T&			_x;
	const Vector<S>&		_h;
    };

  private:
    enum		{DEFAULT_NITER_MAX = 1000};
		 
  public:
    Minimization(S tol = DEFAULT_TOL, int niter_max = DEFAULT_NITER_MAX,
		 int pr = 0)
      :_tol(tol), _niter_max(niter_max), _print(pr)			{}
    
    virtual S		operator ()(const T&)			const	= 0;
    virtual Vector<S>	ngrad(const T& x)			const	= 0;
    virtual T		proceed(const T&, const Vector<S>&)	const	= 0;
    S			minimize(T&)					;
    S			steepest_descent(T&)				;
    S			line_minimize(T&, const Vector<S>&)	const	;

  protected:
    virtual void	update(const T&)				;
    virtual void	print(int, S, const T&)			const	;
    
  private:
    int			near_enough(S, S)			const	;
 
    const S		_tol;
    const int		_niter_max;
    const int		_print;
};

template <class S, class T> inline int
Minimization<S, T>::near_enough(S a, S b) const
{
#define EPS	1.0e-10
    return 2.0*fabs(a - b) <= _tol*(fabs(a) + fabs(b) + EPS);
}
 
}

#endif	/* !__TUVectorPP_h	*/
