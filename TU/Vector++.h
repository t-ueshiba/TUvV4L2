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
  \file		Vector++.h
  \brief	ベクトルと行列およびそれに関連するクラスの定義と実装
*/
#ifndef __TU_VECTORPP_H
#define __TU_VECTORPP_H

#include "TU/Array++.h"
#include <numeric>
#include <cmath>
#include <utility>	// std::declval<T>

namespace TU
{
template <class T, class B=Buf<T> >				class Vector;
template <class T, class B=Buf<T>, class R=Buf<Vector<T> > >	class Matrix;

/************************************************************************
*  class Rotation<T>							*
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
template <class T>
class Rotation
{
  public:
    typedef T	element_type;	//!< 成分の型
    
  public:
    Rotation(size_t p, size_t q, element_type x, element_type y)	;
    Rotation(size_t p, size_t q, element_type theta)			;

  //! p軸を返す．
  /*!
    \return	p軸のindex
  */
    size_t		p()				const	{return _p;}

  //! q軸を返す．
  /*!
    \return	q軸のindex
  */
    size_t		q()				const	{return _q;}

  //! 回転角生成ベクトルの長さを返す．
  /*!
    \return	回転角生成ベクトル(x, y)に対して\f$\sqrt{x^2 + y^2}\f$
  */
    element_type	length()			const	{return _l;}

  //! 回転角のcos値を返す．
  /*!
    \return	回転角のcos値
  */
    element_type	cos()				const	{return _c;}

  //! 回転角のsin値を返す．
  /*!
    \return	回転角のsin値
  */
    element_type	sin()				const	{return _s;}
    
  private:
    const size_t		_p, _q;		// rotation axis
    element_type	_l;		// length of (x, y)
    element_type	_c, _s;		// cos & sin
};

//! 2次元超平面内での回転を生成する
/*!
  \param p	p軸を指定するindex
  \param q	q軸を指定するindex
  \param x	回転角を生成する際のx値
  \param y	回転角を生成する際のy値
		\f[
		  \cos\theta = \frac{x}{\sqrt{x^2+y^2}},{\hskip 1em}
		  \sin\theta = \frac{y}{\sqrt{x^2+y^2}}
		\f]
*/
template <class T> inline
Rotation<T>::Rotation(size_t p, size_t q, element_type x, element_type y)
    :_p(p), _q(q), _l(1), _c(1), _s(0)
{
    const element_type	absx = std::abs(x), absy = std::abs(y);
    _l = (absx > absy ? absx * std::sqrt(1 + (absy*absy)/(absx*absx))
		      : absy * std::sqrt(1 + (absx*absx)/(absy*absy)));
    if (_l != 0)
    {
	_c = x / _l;
	_s = y / _l;
    }
}

//! 2次元超平面内での回転を生成する
/*!
  \param p	p軸を指定するindex
  \param q	q軸を指定するindex
  \param theta	回転角
*/
template <class T> inline
Rotation<T>::Rotation(size_t p, size_t q, element_type theta)
    :_p(p), _q(q), _l(1), _c(std::cos(theta)), _s(std::sin(theta))
{
}

/************************************************************************
*  class Vector<T>							*
************************************************************************/
//! T型の成分を持つベクトルを表すクラス
/*!
  \param T	成分の型
  \param B	バッファ
*/
template <class T, class B>
class Vector : public Array<T, B>
{
  private:
    typedef Array<T, B>					super;
    
  public:
    typedef typename super::element_type		element_type;
    typedef typename super::value_type			value_type;
    typedef typename super::difference_type		difference_type;
    typedef typename super::reference			reference;
    typedef typename super::const_reference		const_reference;
    typedef typename super::iterator			iterator;
    typedef typename super::const_iterator		const_iterator;
    typedef typename super::reverse_iterator		reverse_iterator;
    typedef typename super::const_reverse_iterator	const_reverse_iterator;
  //! 成分の型が等しい3次元ベクトルの型
    typedef Vector<T, Buf<T, 3> >			vector3_type;
  //! 成分の型が等しい3x3行列の型
    typedef Matrix<T, Buf<T, 9>, Buf<Vector<T>, 3> >	matrix33_type;
    
  public:
    Vector()								;
    explicit Vector(size_t d)						;
    Vector(T* p, size_t d)						;
    template <class B2>
    Vector(Vector<T, B2>& v, size_t i, size_t d)			;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Vector(const E& v)							;
    Vector(std::initializer_list<value_type> args)			;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Vector&		operator =(const E& v)				;
    Vector&		operator =(std::
				   initializer_list<value_type> args)	;
    Vector&		operator =(const element_type& c)		;
    
    using		super::data;
    using		super::begin;
    using		super::cbegin;
    using		super::end;
    using		super::cend;
    using		super::rbegin;
    using		super::crbegin;
    using		super::rend;
    using		super::crend;
    using		super::size;
    
    const Vector<T>	operator ()(size_t i, size_t d)		const	;
    Vector<T>		operator ()(size_t i, size_t d)			;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Vector&		operator *=(const E& m)				;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Vector&		operator ^=(const E& v)				;
    T			square()				const	;
    double		length()				const	;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    T			sqdist(const E& v)			const	;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    double		dist(const E& v)			const	;
    Vector&		normalize()					;
    Vector		normal()				const	;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Vector&		solve(const E& m)				;
    matrix33_type	skew()					const	;
    Vector<T>		homogeneous()				const	;
    Vector<T>		inhomogeneous()				const	;
};

//! ベクトルを生成し，全成分を0で初期化する．
template <class T, class B>
Vector<T, B>::Vector()
    :super()
{
}

//! 指定された次元のベクトルを生成し，全成分を0で初期化する．
/*!
  \param d	ベクトルの次元
*/
template <class T, class B> inline
Vector<T, B>::Vector(size_t d)
    :super(d)
{
}

//! 外部記憶領域と次元を指定してベクトルを生成する．
/*!
  \param p	外部記憶領域へのポインタ
  \param d	ベクトルの次元
*/
template <class T, class B> inline
Vector<T, B>::Vector(T* p, size_t d)
    :super(p, d)
{
}

//! 与えられたベクトルと記憶領域を共有する部分ベクトルを生成する．
/*!
  \param v	元のベクトル
  \param i	部分ベクトルの第0成分を指定するindex
  \param d	部分ベクトルの次元
*/
template <class T, class B> template <class B2> inline
Vector<T, B>::Vector(Vector<T, B2>& v, size_t i, size_t d)
    :super(v, i, d)
{
}

//! 他のベクトルと同一成分を持つベクトルを作る(コピーコンストラクタの拡張)．
/*!
  \param v	コピー元ベクトル
*/
template <class T, class B>
template <class E, typename std::enable_if<is_range<E>::value>::type*> inline
Vector<T, B>::Vector(const E& v)
    :super(v)
{
}
    
template <class T, class B> inline
Vector<T, B>::Vector(std::initializer_list<value_type> args)
    :super(args)
{
}
    
//! 他のベクトルを自分に代入する(代入演算子の拡張)．
/*!
  \param v	コピー元ベクトル
  \return	このベクトル
*/

template <class T, class B>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
inline Vector<T, B>&
Vector<T, B>::operator =(const E& v)
{
    super::operator =(v);
    return *this;
}

template <class T, class B> inline Vector<T, B>&
Vector<T, B>::operator =(std::initializer_list<value_type> args)
{
    super::operator =(args);
    return *this;
}

template <class T, class B> inline Vector<T, B>&
Vector<T, B>::operator =(const element_type& c)
{
    super::operator =(c);
    return *this;
}
    
//! このベクトルと記憶領域を共有した部分ベクトルを生成する．
/*!
    \param i	部分ベクトルの第0成分を指定するindex
    \param d	部分ベクトルの次元
    \return	生成された部分ベクトル
*/
template <class T, class B> inline Vector<T>
Vector<T, B>::operator ()(size_t i, size_t d)
{
    return Vector<T>(*this, i, d);
}

//! このベクトルと記憶領域を共有した部分ベクトルを生成する．
/*!
    \param i	部分ベクトルの第0成分を指定するindex
    \param d	部分ベクトルの次元
    \return	生成された部分ベクトル
*/
template <class T, class B> inline const Vector<T>
Vector<T, B>::operator ()(size_t i, size_t d) const
{
    return Vector<T>(const_cast<Vector<T, B>&>(*this), i, d);
}

//! このベクトルの右から行列を掛ける．
/*!
  \param m	掛ける行列
  \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \TUvec{u}{}\TUvec{M}{}\f$
*/
template <class T, class B>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
inline Vector<T, B>&
Vector<T, B>::operator *=(const E& m)
{
    return *this = *this * m;
}

//! このベクトルと他の3次元ベクトルとのベクトル積をとる．
/*!
  \param v	他のベクトル
  \return	このベクトル，すなわち
		\f$\TUvec{u}{}\leftarrow \TUvec{u}{} \times \TUvec{v}{}\f$
*/
template <class T, class B>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
inline Vector<T, B>&
Vector<T, B>::operator ^=(const E& v)
{
    return *this = *this ^ v;
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
    return std::sqrt(double(square()));
}

//! このベクトルと他のベクトルの差の長さの2乗を返す．
/*!
  \param v	比較対象となるベクトル
  \return	ベクトル間の差の2乗，すなわち
		\f$\TUnorm{\TUvec{u}{} - \TUvec{v}{}}^2\f$
*/
template <class T, class B>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
inline T
Vector<T, B>::sqdist(const E& v) const
{
    return Vector(*this - v).square();
}

//! このベクトルと他のベクトルの差の長さを返す．
/*!
  \param v	比較対象となるベクトル
  \return	ベクトル間の差，すなわち
		\f$\TUnorm{\TUvec{u}{} - \TUvec{v}{}}\f$
*/
template <class T, class B>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
inline double
Vector<T, B>::dist(const E& v) const
{
    return std::sqrt(double(sqdist(v)));
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

//! この3次元ベクトルから3x3反対称行列を生成する．
/*!
  \return	生成された反対称行列，すなわち
  \f[
    \TUskew{u}{} \equiv
    \TUbeginarray{ccc}
      & -u_2 & u_1 \\ u_2 & & -u_0 \\ -u_1 & u_0 &
    \TUendarray
  \f]
  \throw std::invalid_argument	3次元ベクトルでない場合に送出
*/
template <class T, class B> Matrix<T, Buf<T, 9>, Buf<Vector<T>, 3> >
Vector<T, B>::skew() const
{
    if (size() != 3)
	throw std::invalid_argument("TU::Vector<T, B>::skew: dimension must be 3");
    matrix33_type	r;
    r[2][1] = (*this)[0];
    r[0][2] = (*this)[1];
    r[1][0] = (*this)[2];
    r[1][2] = -r[2][1];
    r[2][0] = -r[0][2];
    r[0][1] = -r[1][0];
    return r;
}

//! 非同次座標を表すベクトルに対し，値1を持つ成分を最後に付加した同次座標ベクトルを返す．
/*!
  \return	同次化されたベクトル
*/
template <class T, class B> inline Vector<T>
Vector<T, B>::homogeneous() const
{
    Vector<T>	v(size() + 1);
    v(0, size()) = *this;
    v[size()]	 = 1;
    return v;
}

//! 同次座標を表すベクトルに対し，各成分を最後の成分で割った非同次座標ベクトルを返す．
/*!
  \return	非同次化されたベクトル
*/
template <class T, class B> inline Vector<T>
Vector<T, B>::inhomogeneous() const
{
    return (*this)(0, size()-1) / (*this)[size()-1];
}

/************************************************************************
*  class Matrix<T, B, R>						*
************************************************************************/
template <class T>	class BlockDiagonalMatrix;
template <class T>	class TriDiagonal;
template <class T>	class SVDecomposition;

//! T型の成分を持つ行列を表すクラス
/*!
  各行がT型の成分を持つベクトル#TU::Vector<T>になっている．
  \param T	成分の型
  \param B	バッファ
  \param R	行バッファ
*/
template <class T, class B, class R>
class Matrix : public Array2<Vector<T>, B, R>
{
  private:
    typedef Array2<Vector<T>, B, R>			super;
    
  public:
    typedef typename super::element_type		element_type;
    typedef typename super::value_type			value_type;
    typedef typename super::difference_type		difference_type;
    typedef typename super::reference			reference;
    typedef typename super::const_reference		const_reference;
    typedef typename super::iterator			iterator;
    typedef typename super::const_iterator		const_iterator;
    typedef typename super::reverse_iterator		reverse_iterator;
    typedef typename super::const_reverse_iterator	const_reverse_iterator;
  //! 成分の型が等しい3次元ベクトルの型
    typedef Vector<T, Buf<T, 3> >			vector3_type;
  //! 成分の型が等しい4次元ベクトルの型
    typedef Vector<T, Buf<T, 4> >			vector4_type;
  //! 成分の型が等しい2x2行列の型
    typedef Matrix<T, Buf<T, 4>, Buf<Vector<T>, 2> >	matrix22_type;
  //! 成分の型が等しい3x3行列の型
    typedef Matrix<T, Buf<T, 9>, Buf<Vector<T>, 3> >	matrix33_type;
    
  public:
    Matrix()								;
    Matrix(size_t r, size_t c)						;
    Matrix(T* p, size_t r, size_t c)					;
    template <class B2, class R2>
    Matrix(Matrix<T, B2, R2>& m, size_t i, size_t j, size_t r, size_t c);
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Matrix(const E& expr)	:super(expr)				{}
    Matrix(std::initializer_list<value_type> args)			;
    Matrix(const BlockDiagonalMatrix<T>& m)				;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Matrix&		operator =(const E& expr)
			{
			    super::operator =(expr);
			    return *this;
			}
    Matrix&		operator =(std::
				   initializer_list<value_type> arg)	;
    Matrix&		operator =(const BlockDiagonalMatrix<T>& m)	;
    Matrix&		operator =(const element_type& c)		;
    
    using		super::begin;
    using		super::cbegin;
    using		super::end;
    using		super::cend;
    using		super::rbegin;
    using		super::crbegin;
    using		super::rend;
    using		super::crend;
    using		super::size;
    using		super::nrow;
    using		super::ncol;
    using		super::data;

    const Matrix<T>	operator ()(size_t i, size_t j,
				    size_t r, size_t c)		const	;
    Matrix<T>		operator ()(size_t i, size_t j,
				    size_t r, size_t c)			;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Matrix&		operator *=(const E& m)				;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Matrix&		operator ^=(const E& v)				;
    Matrix&		diag(T c)					;
    Matrix<T>		trns()					const	;
    Matrix		inv()					const	;
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Matrix&		solve(const E& m)				;
    T			det()					const	;
    T			det(size_t p, size_t q)			const	;
    T			trace()					const	;
    Matrix		adj()					const	;
    Matrix<T>		pinv(T cndnum=1.0e5)			const	;
    Matrix<T>		eigen(Vector<T>& eval, bool abs=true)	const	;
    Matrix<T>		geigen(const Matrix<T>& BB,
			       Vector<T>& eval, bool abs=true)	const	;
    Matrix		cholesky()				const	;
    Matrix&		normalize()					;
    Matrix&		rotate_from_left(const Rotation<T>& r)		;
    Matrix&		rotate_from_right(const Rotation<T>& r)		;
    T			square()				const	;
    double		length()				const	;
    Matrix&		symmetrize()					;
    Matrix&		antisymmetrize()				;
    void		rot2angle(T& theta_x,
				  T& theta_y, T& theta_z)	const	;
    vector3_type	rot2axis(T& c, T& s)			const	;
    vector3_type	rot2axis()				const	;
    vector4_type	rot2quaternion()			const	;

    static Matrix	I(size_t d)					;
    static matrix22_type
			Rt(T theta)					;
    template <class T2, class B2>
    static matrix33_type
			Rt(const Vector<T2, B2>& n, T c, T s)		;
    template <class T2, class B2>
    static matrix33_type
			Rt(const Vector<T2, B2>& v)			;
};

//! 行列を生成し，全成分を0で初期化する．
template <class T, class B, class R> inline
Matrix<T, B, R>::Matrix()
    :super()
{
}

//! 指定されたサイズの行列を生成し，全成分を0で初期化する．
/*!
  \param r	行列の行数
  \param c	行列の列数
*/
template <class T, class B, class R> inline
Matrix<T, B, R>::Matrix(size_t r, size_t c)
    :super(r, c)
{
}

//! 外部記憶領域とサイズを指定して行列を生成する．
/*!
  \param p	外部記憶領域へのポインタ
  \param r	行列の行数
  \param c	行列の列数
*/
template <class T, class B, class R> inline
Matrix<T, B, R>::Matrix(T* p, size_t r, size_t c)
    :super(p, r, c)
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
template <class T, class B, class R> template <class B2, class R2> inline
Matrix<T, B, R>::Matrix(Matrix<T, B2, R2>& m,
			size_t i, size_t j, size_t r, size_t c)
    :super(m, i, j, r, c)
{
}

template <class T, class B, class R> inline
Matrix<T, B, R>::Matrix(std::initializer_list<value_type> args)
    :super(args)
{
}

template <class T, class B, class R> inline Matrix<T, B, R>&
Matrix<T, B, R>::operator =(std::initializer_list<value_type> args)
{
    super::operator =(args);
    return *this;
}

template <class T, class B, class R> inline Matrix<T, B, R>&
Matrix<T, B, R>::operator =(const element_type& c)
{
    super::operator =(c);
    return *this;
}

//! この行列と記憶領域を共有した部分行列を生成する．
/*!
    \param i	部分行列の左上隅成分となる行を指定するindex
    \param j	部分行列の左上隅成分となる列を指定するindex
    \param r	部分行列の行数
    \param c	部分行列の列数
    \return	生成された部分行列
*/
template <class T, class B, class R> inline Matrix<T>
Matrix<T, B, R>::operator ()(size_t i, size_t j, size_t r, size_t c)
{
    return Matrix<T>(*this, i, j, r, c);
}

//! この行列と記憶領域を共有した部分行列を生成する．
/*!
    \param i	部分行列の左上隅成分となる行を指定するindex
    \param j	部分行列の左上隅成分となる列を指定するindex
    \param r	部分行列の行数
    \param c	部分行列の列数
    \return	生成された部分行列
*/
template <class T, class B, class R> inline const Matrix<T>
Matrix<T, B, R>::operator ()(size_t i, size_t j, size_t r, size_t c) const
{
    return Matrix<T>(const_cast<Matrix<T, B, R>&>(*this), i, j, r, c);
}

//! この行列の右から他の行列を掛ける．
/*!
  \param m	掛ける行列
  \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow \TUvec{A}{}\TUvec{M}{}\f$
*/
template <class T, class B, class R>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
inline Matrix<T, B, R>&
Matrix<T, B, R>::operator *=(const E& m)
{
    return *this = *this * m;
}

//! この?x3行列と3次元ベクトルとのベクトル積をとる．
/*!
  \param v	3次元ベクトル
  \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow(\TUtvec{A}{}\times\TUvec{v}{})^\top\f$
*/
template <class T, class B, class R>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
inline Matrix<T, B, R>&
Matrix<T, B, R>::operator ^=(const E& v)
{
    return *this = *this ^ v;
}

//! この正方行列を全て同一の対角成分値を持つ対角行列にする．
/*!
  \param c	対角成分の値
  \return	この行列，すなわち\f$\TUvec{A}{} \leftarrow \diag(c,\ldots,c)\f$
*/
template <class T, class B, class R> Matrix<T, B, R>&
Matrix<T, B, R>::diag(T c)
{
    super::check_size(ncol());
    *this = element_type(0);
    for (size_t i = 0; i < nrow(); ++i)
	(*this)[i][i] = c;
    return *this;
}

//! この行列の転置行列を返す．
/*!
  \return	転置行列，すなわち\f$\TUtvec{A}{}\f$
*/
template <class T, class B, class R> Matrix<T>
Matrix<T, B, R>::trns() const
{
    Matrix<T> val(ncol(), nrow());
    for (size_t i = 0; i < nrow(); ++i)
	for (size_t j = 0; j < ncol(); ++j)
	    val[j][i] = (*this)[i][j];
    return val;
}

//! この行列の逆行列を返す．
/*!
  \return	逆行列，すなわち\f$\TUinv{A}{}\f$
*/
template <class T, class B, class R> inline Matrix<T, B, R>
Matrix<T, B, R>::inv() const
{
    return I(nrow()).solve(*this);
}

//! この行列の小行列式を返す．
/*!
  \param p	元の行列から取り除く行を指定するindex
  \param q	元の行列から取り除く列を指定するindex
  \return	小行列式，すなわち\f$\det\TUvec{A}{pq}\f$
*/
template <class T, class B, class R> T
Matrix<T, B, R>::det(size_t p, size_t q) const
{
    Matrix<T>		d(nrow()-1, ncol()-1);
    for (size_t i = 0; i < p; ++i)
    {
	for (size_t j = 0; j < q; ++j)
	    d[i][j] = (*this)[i][j];
	for (size_t j = q; j < d.ncol(); ++j)
	    d[i][j] = (*this)[i][j+1];
    }
    for (size_t i = p; i < d.nrow(); ++i)
    {
	for (size_t j = 0; j < q; ++j)
	    d[i][j] = (*this)[i+1][j];
	for (size_t j = q; j < d.ncol(); ++j)
	    d[i][j] = (*this)[i+1][j+1];
    }
    return d.det();
}

//! この正方行列のtraceを返す．
/*!
  \return			trace, すなわち\f$\trace\TUvec{A}{}\f$
  \throw std::invalid_argument	正方行列でない場合に送出
*/
template <class T, class B, class R> T
Matrix<T, B, R>::trace() const
{
    if (nrow() != ncol())
        throw
	  std::invalid_argument("TU::Matrix<T>::trace(): not square matrix!!");
    T	val = 0.0;
    for (size_t i = 0; i < nrow(); ++i)
	val += (*this)[i][i];
    return val;
}

//! この行列の余因子行列を返す．
/*!
  \return	余因子行列，すなわち
		\f$\TUtilde{A}{} = (\det\TUvec{A}{})\TUinv{A}{}\f$
*/
template <class T, class B, class R> Matrix<T, B, R>
Matrix<T, B, R>::adj() const
{
    Matrix<T, B, R>	val(nrow(), ncol());
    for (size_t i = 0; i < val.nrow(); ++i)
	for (size_t j = 0; j < val.ncol(); ++j)
	    val[i][j] = ((i + j) % 2 ? -det(j, i) : det(j, i));
    return val;
}

//! この行列の疑似逆行列を返す．
/*!
  \param cndnum	最大特異値に対する絶対値の割合がこれに達しない基底は無視
  \return	疑似逆行列，すなわち与えられた行列の特異値分解を
		\f$\TUvec{A}{} = \TUvec{V}{}\diag(\sigma_0,\ldots,\sigma_{n-1})
		\TUtvec{U}{}\f$とすると
		\f[
		  \TUvec{u}{0}\sigma_0^{-1}\TUtvec{v}{0} + \cdots +
		  \TUvec{u}{r}\sigma_{r-1}^{-1}\TUtvec{v}{r-1},
		  {\hskip 1em}\mbox{where}{\hskip 0.5em}
		  \TUabs{\sigma_1} > \epsilon\TUabs{\sigma_0},\ldots,
		  \TUabs{\sigma_{r-1}} > \epsilon\TUabs{\sigma_0}
		\f]
*/
template <class T, class B, class R> Matrix<T>
Matrix<T, B, R>::pinv(T cndnum) const
{
    SVDecomposition<T>	svd(*this);
    Matrix<T>		val(svd.ncol(), svd.nrow());
    
    for (size_t i = 0; i < svd.diagonal().size(); ++i)
	if (std::fabs(svd[i]) * cndnum > std::fabs(svd[0]))
	{
	    const Vector<T>	tmp = svd.Ut()[i] / svd[i];
	    val += tmp % svd.Vt()[i];
	}

    return val;
}

//! この対称行列の固有値と固有ベクトルを返す．
/*!
    \param eval	固有値が返される
    \param abs	固有値を絶対値の大きい順に並べるならtrue, 値の大きい順に
		並べるならfalse
    \return	各行が固有ベクトルから成る回転行列，すなわち
		\f[
		  \TUvec{A}{}\TUvec{U}{} =
		  \TUvec{U}{}\diag(\lambda_0,\ldots,\lambda_{n-1}),
		  {\hskip 1em}\mbox{where}{\hskip 0.5em}
		  \TUtvec{U}{}\TUvec{U}{} = \TUvec{I}{n},~\det\TUvec{U}{} = 1
		\f]
		なる\f$\TUtvec{U}{}\f$
*/
template <class T, class B, class R> Matrix<T>
Matrix<T, B, R>::eigen(Vector<T>& eval, bool abs) const
{
    TriDiagonal<T>	tri(*this);

    tri.diagonalize(abs);
    eval = tri.diagonal();

    return tri.Ut();
}

//! この対称行列の一般固有値と一般固有ベクトルを返す．
/*!
    \param BB	もとの行列と同一サイズの正値対称行列
    \param eval	一般固有値が返される
    \param abs	一般固有値を絶対値の大きい順に並べるならtrue, 値の大きい順に
		並べるならfalse
    \return	各行が一般固有ベクトルから成る正則行列
		（ただし直交行列ではない），すなわち
		\f[
		  \TUvec{A}{}\TUvec{U}{} =
		  \TUvec{B}{}\TUvec{U}{}\diag(\lambda_0,\ldots,\lambda_{n-1}),
		  {\hskip 1em}\mbox{where}{\hskip 0.5em}
		  \TUtvec{U}{}\TUvec{B}{}\TUvec{U}{} = \TUvec{I}{n}
		\f]
		なる\f$\TUtvec{U}{}\f$
*/
template <class T, class B, class R> Matrix<T>
Matrix<T, B, R>::geigen(const Matrix<T>& BB, Vector<T>& eval, bool abs) const
{
    Matrix<T>	Ltinv = BB.cholesky().inv(), Linv = Ltinv.trns();
    Matrix<T>	C =  Linv * (*this) * Ltinv;
    Matrix<T>	Ut = C.eigen(eval, abs);
    
    return Ut * Linv;
}

//! この正値対称行列のCholesky分解（上半三角行列）を返す．
/*!
  計算においては，もとの行列の上半部分しか使わない
  \return	\f$\TUvec{A}{} = \TUvec{L}{}\TUtvec{L}{}\f$なる
		\f$\TUtvec{L}{}\f$（上半三角行列）
  \throw std::invalid_argument	正方行列でない場合に送出
  \throw std::runtime_error	正値でない場合に送出
*/
template <class T, class B, class R> Matrix<T, B, R>
Matrix<T, B, R>::cholesky() const
{
    if (nrow() != ncol())
        throw
	    std::invalid_argument("TU::Matrix<T>::cholesky(): not square matrix!!");

    Matrix<T, B, R>	Lt(*this);
    for (size_t i = 0; i < nrow(); ++i)
    {
	T d = Lt[i][i];
	if (d <= 0)
	    throw std::runtime_error("TU::Matrix<T>::cholesky(): not positive definite matrix!!");
	for (size_t j = 0; j < i; ++j)
	    Lt[i][j] = 0;
	Lt[i][i] = d = std::sqrt(d);
	for (size_t j = i + 1; j < ncol(); ++j)
	    Lt[i][j] /= d;
	for (size_t j = i + 1; j < nrow(); ++j)
	    for (size_t k = j; k < ncol(); ++k)
		Lt[j][k] -= (Lt[i][j] * Lt[i][k]);
    }
    
    return Lt;
}

//! この行列のノルムを1に正規化する．
/*!
    \return	この行列，すなわち
		\f$
		  \TUvec{A}{}\leftarrow\frac{\TUvec{A}{}}{\TUnorm{\TUvec{A}{}}}
		\f$
*/
template <class T, class B, class R> Matrix<T, B, R>&
Matrix<T, B, R>::normalize()
{
    T	sum = 0.0;
    for (size_t i = 0; i < nrow(); ++i)
	sum += (*this)[i] * (*this)[i];
    return *this /= std::sqrt(sum);
}

//! この行列の左から（転置された）回転行列を掛ける．
/*!
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow\TUtvec{R}{}\TUvec{A}{}\f$
*/
template <class T, class B, class R> Matrix<T, B, R>&
Matrix<T, B, R>::rotate_from_left(const Rotation<T>& r)
{
    for (size_t j = 0; j < ncol(); ++j)
    {
	const T	tmp = (*this)[r.p()][j];
	
	(*this)[r.p()][j] =  r.cos()*tmp + r.sin()*(*this)[r.q()][j];
	(*this)[r.q()][j] = -r.sin()*tmp + r.cos()*(*this)[r.q()][j];
    }
    return *this;
}

//! この行列の右から回転行列を掛ける．
/*!
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow\TUvec{A}{}\TUvec{R}{}\f$
*/
template <class T, class B, class R> Matrix<T, B, R>&
Matrix<T, B, R>::rotate_from_right(const Rotation<T>& r)
{
    for (size_t i = 0; i < nrow(); ++i)
    {
	const T	tmp = (*this)[i][r.p()];
	
	(*this)[i][r.p()] =  tmp*r.cos() + (*this)[i][r.q()]*r.sin();
	(*this)[i][r.q()] = -tmp*r.sin() + (*this)[i][r.q()]*r.cos();
    }
    return *this;
}

//! この行列の2乗ノルムの2乗を返す．
/*!
    \return	行列の2乗ノルムの2乗，すなわち\f$\TUnorm{\TUvec{A}{}}^2\f$
*/
template <class T, class B, class R> T
Matrix<T, B, R>::square() const
{
    T	val = 0.0;
    for (size_t i = 0; i < nrow(); ++i)
	val += (*this)[i] * (*this)[i];
    return val;
}

//! この行列の2乗ノルムを返す．
/*!
  \return	行列の2乗ノルム，すなわち\f$\TUnorm{\TUvec{A}{}}\f$
*/
template <class T, class B, class R> inline double
Matrix<T, B, R>::length() const
{
    return std::sqrt(double(square()));
}

//! この行列の下半三角部分を上半三角部分にコピーして対称化する．
/*!
    \return	この行列
*/
template <class T, class B, class R> Matrix<T, B, R>&
Matrix<T, B, R>::symmetrize()
{
    for (size_t i = 0; i < nrow(); ++i)
	for (size_t j = 0; j < i; ++j)
	    (*this)[j][i] = (*this)[i][j];
    return *this;
}

//! この行列の下半三角部分の符号を反転し，上半三角部分にコピーして反対称化する．
/*!
    \return	この行列
*/
template <class T, class B, class R> Matrix<T, B, R>&
Matrix<T, B, R>::antisymmetrize()
{
    for (size_t i = 0; i < nrow(); ++i)
    {
	(*this)[i][i] = 0.0;
	for (size_t j = 0; j < i; ++j)
	    (*this)[j][i] = -(*this)[i][j];
    }
    return *this;
}

//! この3次元回転行列から各軸周りの回転角を取り出す．
/*!
  この行列を\f$\TUtvec{R}{}\f$とすると，
  \f[
    \TUvec{R}{} =
    \TUbeginarray{ccc}
      \cos\theta_z & -\sin\theta_z & \\
      \sin\theta_z &  \cos\theta_z & \\
      & & 1
    \TUendarray
    \TUbeginarray{ccc}
       \cos\theta_y & & \sin\theta_y \\
       & 1 & \\
      -\sin\theta_y & & \cos\theta_y
    \TUendarray
    \TUbeginarray{ccc}
      1 & & \\
      & \cos\theta_x & -\sin\theta_x \\
      & \sin\theta_x &  \cos\theta_x
    \TUendarray
  \f]
  なる\f$\theta_x, \theta_y, \theta_z\f$が回転角となる．
 \param theta_x	x軸周りの回転角(\f$ -\pi \le \theta_x \le \pi\f$)を返す．
 \param theta_y	y軸周りの回転角
	(\f$ -\frac{\pi}{2} \le \theta_y \le \frac{\pi}{2}\f$)を返す．
 \param theta_z	z軸周りの回転角(\f$ -\pi \le \theta_z \le \pi\f$)を返す．
 \throw invalid_argument	3次元正方行列でない場合に送出
*/
template <class T, class B, class R> void
Matrix<T, B, R>::rot2angle(T& theta_x, T& theta_y, T& theta_z) const
{
    using namespace	std;
    
    if (nrow() != 3 || ncol() != 3)
	throw invalid_argument("TU::Matrix<T>::rot2angle: input matrix must be 3x3!!");

    if ((*this)[0][0] == 0.0 && (*this)[0][1] == 0.0)
    {
	theta_x = std::atan2(-(*this)[2][1], (*this)[1][1]);
	theta_y = ((*this)[0][2] < 0.0 ? M_PI / 2.0 : -M_PI / 2.0);
	theta_z = 0.0;
    }
    else
    {
	theta_x =  std::atan2((*this)[1][2], (*this)[2][2]);
	theta_y = -std::asin((*this)[0][2]);
	theta_z =  std::atan2((*this)[0][1], (*this)[0][0]);
    }
}

//! この3次元回転行列から回転角と回転軸を取り出す．
/*!
  この行列を\f$\TUtvec{R}{}\f$とすると，
  \f[
    \TUtvec{R}{} \equiv \TUvec{I}{3}\cos\theta
    + \TUvec{n}{}\TUtvec{n}{}(1 - \cos\theta)
    - \TUskew{n}{}\sin\theta
  \f]
  なる\f$\theta~(0 \le \theta \le \pi)\f$と\f$\TUvec{n}{}\f$が
  それぞれ回転角と回転軸となる．
 \param c	回転角のcos値，すなわち\f$\cos\theta\f$を返す．
 \param s	回転角のsin値，すなわち\f$\sin\theta (\ge 0)\f$を返す．
 \return	回転軸を表す3次元単位ベクトル，すなわち\f$\TUvec{n}{}\f$
 \throw std::invalid_argument	3x3行列でない場合に送出
*/
template <class T, class B, class R> Vector<T, Buf<T, 3> >
Matrix<T, B, R>::rot2axis(T& c, T& s) const
{
    if (nrow() != 3 || ncol() != 3)
	throw std::invalid_argument("TU::Matrix<T>::rot2axis: input matrix must be 3x3!!");

  // Compute cosine and sine of rotation angle.
    const T	trace = (*this)[0][0] + (*this)[1][1] + (*this)[2][2];
    c = T(0.5) * (trace - T(1));
    s = T(0.5) * std::sqrt((T(1) + trace)*(T(3) - trace));

  // Compute rotation axis.
    vector3_type	n;
    n[0] = (*this)[1][2] - (*this)[2][1];
    n[1] = (*this)[2][0] - (*this)[0][2];
    n[2] = (*this)[0][1] - (*this)[1][0];
    n.normalize();

    return n;
}

//! この3次元回転行列から回転角と回転軸を取り出す．
/*!
  この行列を\f$\TUtvec{R}{}\f$とすると，
  \f[
    \TUtvec{R}{} \equiv \TUvec{I}{3}\cos\theta
    + \TUvec{n}{}\TUtvec{n}{}(1 - \cos\theta)
    - \TUskew{n}{}\sin\theta
  \f]
  なる\f$\theta~(0 \le \theta \le \pi)\f$と\f$\TUvec{n}{}\f$が
  それぞれ回転角と回転軸となる．
 \return			回転角と回転軸を表す3次元ベクトル，すなわち
				\f$\theta\TUvec{n}{}\f$
 \throw invalid_argument	3x3行列でない場合に送出
*/
template <class T, class B, class R> Vector<T, Buf<T, 3> >
Matrix<T, B, R>::rot2axis() const
{
    if (nrow() != 3 || ncol() != 3)
	throw std::invalid_argument("TU::Matrix<T>::rot2axis: input matrix must be 3x3!!");

    const T	trace = (*this)[0][0] + (*this)[1][1] + (*this)[2][2],
		s2 = std::sqrt((T(1) + trace)*(T(3) - trace));	// 2*sin
    if (s2 + T(1) == T(1))			// sin << 1 ?
	return vector3_type();			// zero vector
    
    vector3_type	axis;
    axis[0] = (*this)[1][2] - (*this)[2][1];
    axis[1] = (*this)[2][0] - (*this)[0][2];
    axis[2] = (*this)[0][1] - (*this)[1][0];

    return axis * (std::atan2(s2, trace - T(1)) / s2);
}

//! この3次元回転行列から四元数を取り出す．
/*!
  この行列を\f$\TUtvec{R}{}\f$とすると，
  \f[
    \TUtvec{R}{} \equiv \TUvec{I}{3}\cos\theta
    + \TUvec{n}{}\TUtvec{n}{}(1 - \cos\theta)
    - \TUskew{n}{}\sin\theta
  \f]
  なる\f$\theta~(0 \le \theta \le \pi)\f$と\f$\TUvec{n}{}\f$に対して，四元数は
  \f[
    \TUvec{q}{} \equiv
    \TUbeginarray{c}
      \cos\frac{\theta}{2} \\ \TUvec{n}{}\sin\frac{\theta}{2}
    \TUendarray
  \f]
  と定義される．
 \return			四元数を表す4次元単位ベクトル
 \throw invalid_argument	3x3行列でない場合に送出
*/
template <class T, class B, class R> Vector<T, Buf<T, 4> >
Matrix<T, B, R>::rot2quaternion() const
{
    if (nrow() != 3 || ncol() != 3)
	throw std::invalid_argument("TU::Matrix<T>::rot2quaternion: input matrix must be 3x3!!");

    vector4_type	q;
    q[0] = T(0.5) * std::sqrt(trace() + T(1));
    if (q[0] + T(1) == T(1))	// q[0] << 1 ?
    {
	Vector<T>	eval;
	q(1, 3) = eigen(eval, false)[0];
    }
    else
    {
	const Matrix<T>&	S = trns() - *this;
	q[1] = T(0.25) * S[2][1] / q[0];
	q[2] = T(0.25) * S[0][2] / q[0];
	q[3] = T(0.25) * S[1][0] / q[0];
    }

    return q;
}

//! 単位正方行列を生成する．
/*!
  \param d	単位正方行列の次元
  \return	単位正方行列
*/
template <class T, class B, class R> inline Matrix<T, B, R>
Matrix<T, B, R>::I(size_t d)
{
    return Matrix<T, B, R>(d, d).diag(1.0);
}

template <class T, class B, class R>
inline typename Matrix<T, B, R>::matrix22_type
Matrix<T, B, R>::Rt(T theta)
{
    matrix22_type	Qt;
    Qt[0][0] = Qt[1][1] = std::cos(theta);
    Qt[0][1] = std::sin(theta);
    Qt[1][0] = -Qt[0][1];

    return Qt;
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
template <class T, class B, class R> template <class T2, class B2>
inline typename Matrix<T, B, R>::matrix33_type
Matrix<T, B, R>::Rt(const Vector<T2, B2>& n, T c, T s)
{
    if (n.size() != 3)
	throw std::invalid_argument("TU::Matrix<T, B, R>::Rt: dimension of the argument \'n\' must be 3");
    matrix33_type	Qt = n % n;
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
  \param v	回転角と回転軸を表す3次元ベクトルまたは四元数を表す4次元単位ベクトル
  \return	生成された回転行列，すなわち3次元ベクトルの場合は
		\f[
		  \TUtvec{R}{} \equiv \TUvec{I}{3}\cos\theta
		  + \TUvec{n}{}\TUtvec{n}{}(1 - \cos\theta)
		  - \TUskew{n}{}\sin\theta,
		  {\hskip 1em}\mbox{where}{\hskip 0.5em}
		  \theta \equiv \TUnorm{\TUvec{v}{}},~
		  \TUvec{n}{} \equiv \frac{\TUvec{v}{}}{\TUnorm{\TUvec{v}{}}}
		\f]
		4次元単位ベクトルの場合は
		\f[
		  \TUtvec{R}{} \equiv
		  \TUvec{I}{3}(q_0^2 - \TUtvec{q}{}\TUvec{q}{})
		  + 2\TUvec{q}{}\TUtvec{q}{}
		  - 2q_0\TUskew{q}{},
		  {\hskip 1em}\mbox{where}{\hskip 0.5em}
		  q_0 \equiv v_0,~
		  \TUvec{q}{} \equiv [v_1,~v_2,~v_3]^\top
		\f]
*/
template <class T, class B, class R> template <class T2, class B2>
typename Matrix<T, B, R>::matrix33_type
Matrix<T, B, R>::Rt(const Vector<T2, B2>& v)
{
    if (v.size() == 4)			// quaternion ?
    {
	const T		q0 = v[0];
	vector3_type	q;
	q[0] = v[1];
	q[1] = v[2];
	q[2] = v[3];
	matrix33_type	Qt;
	Qt = (2.0 * q) % q;
	const T		c = q0*q0 - q.square();
	Qt[0][0] += c;
	Qt[1][1] += c;
	Qt[2][2] += c;
	q *= (2.0 * q0);
	Qt[0][1] += q[2];
	Qt[0][2] -= q[1];
	Qt[1][0] -= q[2];
	Qt[1][2] += q[0];
	Qt[2][0] += q[1];
	Qt[2][1] -= q[0];

	return Qt;
    }

    const T	theta = v.length();
    if (theta + T(1) == T(1))		// theta << 1 ?
	return I(3);
    else
    {
	T	c = std::cos(theta), s = std::sin(theta);
	return Rt(vector3_type(v / theta), c, s);
    }
}

/************************************************************************
*  class LUDecomposition<T>						*
************************************************************************/
//! 正方行列のLU分解を表すクラス
template <class T>
class LUDecomposition : private Array2<Vector<T> >
{
  private:
    typedef T						element_type;
    typedef Array2<Vector<T> >				super;
    
  public:
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    LUDecomposition(const E& m)				;

    template <class T2, class B2>
    void	substitute(Vector<T2, B2>& b)	const	;

  //! もとの正方行列の行列式を返す．
  /*!
    \return	もとの正方行列の行列式
  */
    T		det()				const	{return _det;}
    
  private:
    using	super::nrow;
    using	super::ncol;
    
    Array<int>	_index;
    T		_det;
};

//! 与えられた正方行列のLU分解を生成する．
/*!
 \param m			LU分解する正方行列
 \throw std::invalid_argument	mが正方行列でない場合に送出
*/
template <class T>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
LUDecomposition<T>::LUDecomposition(const E& m)
    :super(m), _index(ncol()), _det(1.0)
{
    using namespace	std;
    
    if (nrow() != ncol())
        throw invalid_argument("TU::LUDecomposition<T>::LUDecomposition: not square matrix!!");

    for (size_t j = 0; j < ncol(); ++j)	// initialize column index
	_index[j] = j;			// for explicit pivotting

    Vector<T>	scale(ncol());
    for (size_t j = 0; j < ncol(); ++j)	// find maximum abs. value in each col.
    {					// for implicit pivotting
	T max = 0.0;

	for (size_t i = 0; i < nrow(); ++i)
	{
	    const T tmp = std::fabs((*this)[i][j]);
	    if (tmp > max)
		max = tmp;
	}
	scale[j] = (max != 0.0 ? 1.0 / max : 1.0);
    }

    for (size_t i = 0; i < nrow(); ++i)
    {
	for (size_t j = 0; j < i; ++j)		// left part (j < i)
	{
	    T& sum = (*this)[i][j];
	    for (size_t k = 0; k < j; ++k)
		sum -= (*this)[i][k] * (*this)[k][j];
	}

	size_t	jmax = i;
	T	max = 0.0;
	for (size_t j = i; j < ncol(); ++j)  // diagonal and right part (i <= j)
	{
	    T& sum = (*this)[i][j];
	    for (size_t k = 0; k < i; ++k)
		sum -= (*this)[i][k] * (*this)[k][j];
	    const T tmp = std::fabs(sum) * scale[j];
	    if (tmp >= max)
	    {
		max  = tmp;
		jmax = j;
	    }
	}
	if (jmax != i)			// pivotting required ?
	{
	    for (size_t k = 0; k < nrow(); ++k)	// swap i-th and jmax-th column
		swap((*this)[k][i], (*this)[k][jmax]);
	    swap(_index[i], _index[jmax]);	// swap column index
	    swap(scale[i], scale[jmax]);	// swap colum-wise scale factor
	    _det = -_det;
	}

	_det *= (*this)[i][i];

	if ((*this)[i][i] == 0.0)	// singular matrix ?
	    break;

	for (size_t j = i + 1; j < nrow(); ++j)
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
    if (b.size() != ncol())
	throw std::invalid_argument("TU::LUDecomposition<T>::substitute: Dimension of given vector is not equal to mine!!");
    
    Vector<T2, B2>	tmp(b);
    for (size_t j = 0; j < b.size(); ++j)
	b[j] = tmp[_index[j]];

    for (size_t j = 0; j < b.size(); ++j)	// forward substitution
	for (size_t i = 0; i < j; ++i)
	    b[j] -= b[i] * (*this)[i][j];
    for (size_t j = b.size(); j-- > 0; )	// backward substitution
    {
	for (size_t i = b.size(); --i > j; )
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
template <class T, class B>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
inline Vector<T, B>&
Vector<T, B>::solve(const E& m)
{
    LUDecomposition<T>(m).substitute(*this);
    return *this;
}

//! 連立1次方程式を解く．
/*!
  \param m	正則な正方行列
  \return	\f$\TUvec{A}{} = \TUvec{X}{}\TUvec{M}{}\f$
		の解を納めたこの行列，すなわち
		\f$\TUvec{A}{} \leftarrow \TUvec{A}{}\TUinv{M}{}\f$
*/
template <class T, class B, class R>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
Matrix<T, B, R>&
Matrix<T, B, R>::solve(const E& m)
{
    LUDecomposition<T>	lu(m);
    
    for (size_t i = 0; i < nrow(); ++i)
	lu.substitute((*this)[i]);
    return *this;
}

//! この行列の行列式を返す．
/*!
  \return	行列式，すなわち\f$\det\TUvec{A}{}\f$
*/
template <class T, class B, class R> inline T
Matrix<T, B, R>::det() const
{
    return LUDecomposition<T>(*this).det();
}

/************************************************************************
*  class Householder<T>							*
************************************************************************/
template <class T>	class QRDecomposition;
template <class T>	class BiDiagonal;

//! Householder変換を表すクラス
template <class T>
class Householder : public Matrix<T>
{
  private:
    typedef Matrix<T>						super;
    
  public:
    typedef T							element_type;
    
  private:
    Householder(size_t dd, size_t d)
	:super(dd, dd), _d(d), _sigma(Matrix<T>::nrow())	{}
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    Householder(const E& a, size_t d)				;

    using		super::size;
    
    void		apply_from_left(Matrix<T>& a, size_t m)	;
    void		apply_from_right(Matrix<T>& a, size_t m);
    void		apply_from_both(Matrix<T>& a, size_t m)	;
    void		make_transformation()			;
    const Vector<T>&	sigma()				const	{return _sigma;}
    Vector<T>&		sigma()					{return _sigma;}
    bool		sigma_is_zero(size_t m, T comp)	const	;

  private:
    const size_t	_d;		// deviation from diagonal element
    Vector<T>		_sigma;

    friend class	QRDecomposition<T>;
    friend class	TriDiagonal<T>;
    friend class	BiDiagonal<T>;
};

template <class T>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
Householder<T>::Householder(const E& a, size_t d)
    :super(a), _d(d), _sigma(size())
{
    if (a.size() != a.ncol())
	throw std::invalid_argument("TU::Householder<T>::Householder: Given matrix must be square !!");
}

template <class T> void
Householder<T>::apply_from_left(Matrix<T>& a, size_t m)
{
    if (a.nrow() < size())
	throw std::invalid_argument("TU::Householder<T>::apply_from_left: # of rows of given matrix is smaller than my dimension !!");
    
    T	scale = 0.0;
    for (size_t i = m+_d; i < size(); ++i)
	scale += std::fabs(a[i][m]);
	
    if (scale != 0.0)
    {
	T	h = 0.0;
	for (size_t i = m+_d; i < size(); ++i)
	{
	    a[i][m] /= scale;
	    h += a[i][m] * a[i][m];
	}

	const T	s = (a[m+_d][m] > 0.0 ? std::sqrt(h) : -std::sqrt(h));
	h	     += s * a[m+_d][m];			// H = u^2 / 2
	a[m+_d][m]   += s;				// m-th col <== u
	    
	for (size_t j = m+1; j < a.ncol(); ++j)
	{
	    T	p = 0.0;
	    for (size_t i = m+_d; i < size(); ++i)
		p += a[i][m] * a[i][j];
	    p /= h;					// p[j] (p' = u'A / H)
	    for (size_t i = m+_d; i < size(); ++i)
		a[i][j] -= a[i][m] * p;			// A = A - u*p'
	    a[m+_d][j] = -a[m+_d][j];
	}
	    
	for (size_t i = m+_d; i < size(); ++i)
	    (*this)[m][i] = scale * a[i][m];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::apply_from_right(Matrix<T>& a, size_t m)
{
    if (a.ncol() < size())
	throw std::invalid_argument("Householder<T>::apply_from_right: # of column of given matrix is smaller than my dimension !!");
    
    T	scale = 0.0;
    for (size_t j = m+_d; j < size(); ++j)
	scale += std::fabs(a[m][j]);
	
    if (scale != 0.0)
    {
	T	h = 0.0;
	for (size_t j = m+_d; j < size(); ++j)
	{
	    a[m][j] /= scale;
	    h += a[m][j] * a[m][j];
	}

	const T	s = (a[m][m+_d] > 0.0 ? std::sqrt(h) : -std::sqrt(h));
	h	     += s * a[m][m+_d];			// H = u^2 / 2
	a[m][m+_d]   += s;				// m-th row <== u

	for (size_t i = m+1; i < a.nrow(); ++i)
	{
	    T	p = 0.0;
	    for (size_t j = m+_d; j < size(); ++j)
		p += a[i][j] * a[m][j];
	    p /= h;					// p[i] (p = Au / H)
	    for (size_t j = m+_d; j < size(); ++j)
		a[i][j] -= p * a[m][j];			// A = A - p*u'
	    a[i][m+_d] = -a[i][m+_d];
	}
	    
	for (size_t j = m+_d; j < size(); ++j)
	    (*this)[m][j] = scale * a[m][j];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::apply_from_both(Matrix<T>& a, size_t m)
{
    Vector<T>	u = a[m](m+_d, a.ncol()-m-_d);
    T		scale = 0.0;
    for (size_t j = 0; j < u.size(); ++j)
	scale += std::fabs(u[j]);
	
    if (scale != 0.0)
    {
	u /= scale;

	T		h = u * u;
	const T	s = (u[0] > 0.0 ? std::sqrt(h) : -std::sqrt(h));
	h	     += s * u[0];			// H = u^2 / 2
	u[0]	     += s;				// m-th row <== u

	Matrix<T>	A = a(m+_d, m+_d, a.nrow()-m-_d, a.ncol()-m-_d);
	Vector<T>	p = _sigma(m+_d, size()-m-_d);
	for (size_t i = 0; i < A.nrow(); ++i)
	    p[i] = (A[i] * u) / h;			// p = Au / H

	const T	k = (u * p) / (h + h);		// K = u*p / 2H
	for (size_t i = 0; i < A.nrow(); ++i)
	{				// m-th col of 'a' is used as 'q'
	    a[m+_d+i][m] = p[i] - k * u[i];		// q = p - Ku
	    for (size_t j = 0; j <= i; ++j)		// A = A - uq' - qu'
		A[j][i] = (A[i][j] -= (u[i]*a[m+_d+j][m] + a[m+_d+i][m]*u[j]));
	}
	for (size_t j = 1; j < A.nrow(); ++j)
	    A[j][0] = A[0][j] = -A[0][j];

	for (size_t j = m+_d; j < a.ncol(); ++j)
	    (*this)[m][j] = scale * a[m][j];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::make_transformation()
{
    for (size_t m = size(); m-- > 0; )
    {
	for (size_t i = m+1; i < size(); ++i)
	    (*this)[i][m] = 0.0;

	if (_sigma[m] != 0.0)
	{
	    for (size_t i = m+1; i < size(); ++i)
	    {
		T	g = 0.0;
		for (size_t j = m+1; j < size(); ++j)
		    g += (*this)[i][j] * (*this)[m-_d][j];
		g /= (_sigma[m] * (*this)[m-_d][m]);	// g[i] (g = Uu / H)
		for (size_t j = m; j < size(); ++j)
		    (*this)[i][j] -= g * (*this)[m-_d][j];	// U = U - gu'
	    }
	    for (size_t j = m; j < size(); ++j)
		(*this)[m][j] = (*this)[m-_d][j] / _sigma[m];
	    (*this)[m][m] -= 1.0;
	}
	else
	{
	    for (size_t j = m+1; j < size(); ++j)
		(*this)[m][j] = 0.0;
	    (*this)[m][m] = 1.0;
	}
    }
}

template <class T> bool
Householder<T>::sigma_is_zero(size_t m, T comp) const
{
    return (T(std::fabs(_sigma[m])) + comp == comp);
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
  private:
    typedef Matrix<T>					super;
    
  public:
    typedef T						element_type;
    
  public:
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    QRDecomposition(const E& m)				;

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
    using		super::nrow;
    using		super::ncol;
    
    Householder<T>	_Qt;			// rotation matrix
};

//! 与えられた一般行列のQR分解を生成する．
/*!
 \param m	QR分解する一般行列
*/
template <class T>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
QRDecomposition<T>::QRDecomposition(const E& m)
    :super(m), _Qt(m.ncol(), 0)
{
    size_t	n = std::min(nrow(), ncol());
    for (size_t j = 0; j < n; ++j)
	_Qt.apply_from_right(*this, j);
    _Qt.make_transformation();
    for (size_t i = 0; i < n; ++i)
    {
	(*this)[i][i] = _Qt.sigma()[i];
	for (size_t j = i + 1; j < ncol(); ++j)
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
    typedef T		element_type;	//!< 成分の型
    
  public:
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    TriDiagonal(const E& a)				;

  //! 3重対角化される対称行列の次元(= 行数 = 列数)を返す．
  /*!
    \return	対称行列の次元
  */
    size_t		size()			const	{return _Ut.nrow();}

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

    void		diagonalize(bool abs=true)	;
    
  private:
    enum		{NITER_MAX = 30};

    bool		off_diagonal_is_zero(size_t n)		const	;
    void		initialize_rotation(size_t m, size_t n,
					    T& x, T& y)		const	;
    
    Householder<T>	_Ut;
    Vector<T>		_diagonal;
    Vector<T>&		_off_diagonal;
};

//! 与えられた対称行列を3重対角化する．
/*!
  \param a			3重対角化する対称行列
  \throw std::invalid_argument	aが正方行列でない場合に送出
*/
template <class T>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
TriDiagonal<T>::TriDiagonal(const E& a)
    :_Ut(a, 1), _diagonal(_Ut.nrow()), _off_diagonal(_Ut.sigma())
{
    if (_Ut.nrow() != _Ut.ncol())
        throw std::invalid_argument("TU::TriDiagonal<T>::TriDiagonal: not square matrix!!");

    for (size_t m = 0; m < size(); ++m)
    {
	_Ut.apply_from_both(_Ut, m);
	_diagonal[m] = _Ut[m][m];
    }

    _Ut.make_transformation();
}

//! 3重対角行列を対角化する（固有値，固有ベクトルの計算）．
/*!
  対角成分は固有値となり，\f$\TUtvec{U}{}\f$の各行は固有ベクトルを与える．
  \throw std::runtime_error	指定した繰り返し回数を越えた場合に送出
  \param abs	固有値をその絶対値の大きい順に並べるのであればtrue,
		その値の大きい順に並べるのであればfalse
*/ 
template <class T> void
TriDiagonal<T>::diagonalize(bool abs)
{
    using namespace	std;
    
    for (size_t n = size(); n-- > 0; )
    {
	int	niter = 0;
	
#ifdef TU_DEBUG
	cerr << "******** n = " << n << " ********" << endl;
#endif
	while (!off_diagonal_is_zero(n))
	{					// n > 0 here
	    if (niter++ > NITER_MAX)
		throw runtime_error("TU::TriDiagonal::diagonalize(): Number of iteration exceeded maximum value!!");

	  /* Find first m (< n) whose off-diagonal element is 0 */
	    size_t	m = n;
	    while (!off_diagonal_is_zero(--m))	// 0 <= m < n < size() here
	    {
	    }

	  /* Set x and y which determine initial(i = m+1) plane rotation */
	    T	x, y;
	    initialize_rotation(m, n, x, y);
	  /* Apply rotation P(i-1, i) for each i (i = m+1, n+2, ... , n) */
	    for (size_t i = m; ++i <= n; )
	    {
		Rotation<T>	rot(i-1, i, x, y);
		
		_Ut.rotate_from_left(rot);

		if (i > m+1)
		    _off_diagonal[i-1] = rot.length();
		const T w = _diagonal[i] - _diagonal[i-1];
		const T d = rot.sin()*(rot.sin()*w
			       + 2.0*rot.cos()*_off_diagonal[i]);
		_diagonal[i-1]	 += d;
		_diagonal[i]	 -= d;
		_off_diagonal[i] += rot.sin()*(rot.cos()*w
				  - 2.0*rot.sin()*_off_diagonal[i]);
		if (i < n)
		{
		    x = _off_diagonal[i];
		    y = rot.sin()*_off_diagonal[i+1];
		    _off_diagonal[i+1] *= rot.cos();
		}
	    }
#ifdef TU_DEBUG
	    cerr << "  niter = " << niter << ": " << off_diagonal();
#endif	    
	}
    }

  // Sort eigen values and eigen vectors.
    if (abs)
    {
	for (size_t m = 0; m < size(); ++m)
	    for (size_t n = m+1; n < size(); ++n)
		if (std::fabs(_diagonal[n]) >
		    std::fabs(_diagonal[m]))			// abs. values
		{
		    swap(_diagonal[m], _diagonal[n]);
		    for (size_t j = 0; j < size(); ++j)
		    {
			const T	tmp = _Ut[m][j];
			_Ut[m][j] = _Ut[n][j];
			_Ut[n][j] = -tmp;
		    }
		}
    }
    else
    {
	for (size_t m = 0; m < size(); ++m)
	    for (size_t n = m+1; n < size(); ++n)
		if (_diagonal[n] > _diagonal[m])		// raw values
		{
		    swap(_diagonal[m], _diagonal[n]);
		    for (size_t j = 0; j < size(); ++j)
		    {
			const T	tmp = _Ut[m][j];
			_Ut[m][j] = _Ut[n][j];
			_Ut[n][j] = -tmp;
		    }
		}
    }
}

template <class T> bool
TriDiagonal<T>::off_diagonal_is_zero(size_t n) const
{
    return (n == 0 || _Ut.sigma_is_zero(n, std::fabs(_diagonal[n-1]) +
					   std::fabs(_diagonal[n])));
}

template <class T> void
TriDiagonal<T>::initialize_rotation(size_t m, size_t n, T& x, T& y) const
{
    const T	g    = (_diagonal[n] - _diagonal[n-1]) / (2.0*_off_diagonal[n]),
		absg = std::fabs(g),
		gg1  = (absg > 1.0 ?
			absg * std::sqrt(1.0 + (1.0/absg)*(1.0/absg)) :
			std::sqrt(1.0 + absg*absg)),
		t    = (g > 0.0 ? g + gg1 : g - gg1);
    x = _diagonal[m] - _diagonal[n] - _off_diagonal[n]/t;
  //x = _diagonal[m];					// without shifting
    y = _off_diagonal[m+1];
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
    typedef T					element_type;	//!< 成分の型
    
  public:
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    BiDiagonal(const E& a)			;

  //! 2重対角化される行列の行数を返す．
  /*!
    \return	行列の行数
  */
    size_t		nrow()		const	{return _Vt.nrow();}

  //! 2重対角化される行列の列数を返す．
  /*!
    \return	行列の列数
  */
    size_t		ncol()		const	{return _Ut.nrow();}

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
    
    bool		diagonal_is_zero(size_t n)		const	;
    bool		off_diagonal_is_zero(size_t n)		const	;
    void		initialize_rotation(size_t m, size_t n,
					    T& x, T& y)		const	;

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
template <class T>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
BiDiagonal<T>::BiDiagonal(const E& a)
    :_Dt((a.size() < a.ncol() ? a.ncol() : a.size()), 0),
     _Et((a.size() < a.ncol() ? a.size() : a.ncol()), 1),
     _diagonal(_Dt.sigma()), _off_diagonal(_Et.sigma()), _anorm(0),
     _Ut(a.size() < a.ncol() ? _Dt : _Et),
     _Vt(a.size() < a.ncol() ? _Et : _Dt)
{
    detail::argument_t<E, true>	A(a);
    
    if (nrow() < ncol())
	for (size_t i = 0; i < nrow(); ++i)
	    for (size_t j = 0; j < ncol(); ++j)
		_Dt[i][j] = A[i][j];
    else
	for (size_t i = 0; i < nrow(); ++i)
	    for (size_t j = 0; j < ncol(); ++j)
		_Dt[j][i] = A[i][j];

  /* Householder reduction to bi-diagonal (off-diagonal in lower part) form */
    for (size_t m = 0; m < _Et.size(); ++m)
    {
	_Dt.apply_from_right(_Dt, m);
	_Et.apply_from_left(_Dt, m);
    }

    _Dt.make_transformation();	// Accumulate right-hand transformation: V
    _Et.make_transformation();	// Accumulate left-hand transformation: U

    for (size_t m = 0; m < _Et.size(); ++m)
    {
	T	anorm = std::fabs(_diagonal[m]) + std::fabs(_off_diagonal[m]);
	if (anorm > _anorm)
	    _anorm = anorm;
    }
}

//! 2重対角行列を対角化する（特異値分解）．
/*!
  対角成分は特異値となり，\f$\TUtvec{U}{}\f$と\f$\TUtvec{V}{}\f$
  の各行はそれぞれ右特異ベクトルと左特異ベクトルを与える．
  \throw std::runtime_error	指定した繰り返し回数を越えた場合に送出
*/ 
template <class T> void
BiDiagonal<T>::diagonalize()
{
    using namespace	std;
    
    for (size_t n = _Et.size(); n-- > 0; )
    {
	size_t	niter = 0;
	
#ifdef TU_DEBUG
	cerr << "******** n = " << n << " ********" << endl;
#endif
	while (!off_diagonal_is_zero(n))	// n > 0 here
	{
	    if (niter++ > NITER_MAX)
		throw runtime_error("TU::BiDiagonal::diagonalize(): Number of iteration exceeded maximum value");
	    
	  /* Find first m (< n) whose off-diagonal element is 0 */
	    size_t m = n;
	    do
	    {
		if (diagonal_is_zero(m-1))
		{ // If _diagonal[m-1] is zero, make _off_diagonal[m] zero.
		    T	x = _diagonal[m], y = _off_diagonal[m];
		    _off_diagonal[m] = 0.0;
		    for (size_t i = m; i <= n; ++i)
		    {
			Rotation<T>	rotD(m-1, i, x, -y);

			_Dt.rotate_from_left(rotD);
			
			_diagonal[i] = -y*rotD.sin()
				     + _diagonal[i]*rotD.cos();
			if (i < n)
			{
			    x = _diagonal[i+1];
			    y = _off_diagonal[i+1]*rotD.sin();
			    _off_diagonal[i+1] *= rotD.cos();
			}
		    }
		    break;	// if _diagonal[n-1] is zero, m == n here.
		}
	    } while (!off_diagonal_is_zero(--m)); // 0 <= m < n < nrow() here.
	    if (m == n)
		break;		// _off_diagonal[n] has been made 0. Retry!

	  /* Set x and y which determine initial(i = m+1) plane rotation */
	    T	x, y;
	    initialize_rotation(m, n, x, y);
#ifdef TU_DEBUG
	    cerr << "--- m = " << m << ", n = " << n << "---"
		 << endl;
	    cerr << "  diagonal:     " << diagonal();
	    cerr << "  off-diagonal: " << off_diagonal();
#endif
	  /* Apply rotation P(i-1, i) for each i (i = m+1, n+2, ... , n) */
	    for (size_t i = m; ++i <= n; )
	    {
	      /* Apply rotation from left */
		Rotation<T>	rotE(i-1, i, x, y);
		
		_Et.rotate_from_left(rotE);

		if (i > m+1)
		    _off_diagonal[i-1] = rotE.length();
		T	tmp = _diagonal[i-1];
		_diagonal[i-1]	 =  rotE.cos()*tmp
				 +  rotE.sin()*_off_diagonal[i];
		_off_diagonal[i] = -rotE.sin()*tmp
				 +  rotE.cos()*_off_diagonal[i];
		if (diagonal_is_zero(i))
		    break;		// No more Given's rotation needed.
		y		 =  rotE.sin()*_diagonal[i];
		_diagonal[i]	*=  rotE.cos();

		x = _diagonal[i-1];
		
	      /* Apply rotation from right to recover bi-diagonality */
		Rotation<T>	rotD(i-1, i, x, y);

		_Dt.rotate_from_left(rotD);

		_diagonal[i-1] = rotD.length();
		tmp = _off_diagonal[i];
		_off_diagonal[i] =  tmp*rotD.cos() + _diagonal[i]*rotD.sin();
		_diagonal[i]	 = -tmp*rotD.sin() + _diagonal[i]*rotD.cos();
		if (i < n)
		{
		    if (off_diagonal_is_zero(i+1))
			break;		// No more Given's rotation needed.
		    y		        = _off_diagonal[i+1]*rotD.sin();
		    _off_diagonal[i+1] *= rotD.cos();

		    x		        = _off_diagonal[i];
		}
	    }
#ifdef TU_DEBUG
	    cerr << "  niter = " << niter << ": " << off_diagonal();
#endif
	}
    }

    for (size_t m = 0; m < _Et.size(); ++m)  // sort singular values and vectors
	for (size_t n = m+1; n < _Et.size(); ++n)
	    if (std::fabs(_diagonal[n]) > std::fabs(_diagonal[m]))
	    {
		swap(_diagonal[m], _diagonal[n]);
		for (size_t j = 0; j < _Et.size(); ++j)
		{
		    const T	tmp = _Et[m][j];
		    _Et[m][j] = _Et[n][j];
		    _Et[n][j] = -tmp;
		}
		for (size_t j = 0; j < _Dt.size(); ++j)
		{
		    const T	tmp = _Dt[m][j];
		    _Dt[m][j] = _Dt[n][j];
		    _Dt[n][j] = -tmp;
		}
	    }

    size_t l = min(_Dt.size() - 1, _Et.size());	// last index
    for (size_t m = 0; m < l; ++m)	// ensure positivity of all singular
	if (_diagonal[m] < 0.0)		// values except for the last one.
	{
	    _diagonal[m] = -_diagonal[m];
	    for (size_t j = 0; j < _Et.size(); ++j)
		_Et[m][j] = -_Et[m][j];

	    if (l < _Et.size())
	    {
		_diagonal[l] = -_diagonal[l];
		for (size_t j = 0; j < _Et.size(); ++j)
		    _Et[l][j] = -_Et[l][j];
	    }
	}
}

template <class T> bool
BiDiagonal<T>::diagonal_is_zero(size_t n) const
{
    return _Dt.sigma_is_zero(n, _anorm);
}

template <class T> bool
BiDiagonal<T>::off_diagonal_is_zero(size_t n) const
{
    return _Et.sigma_is_zero(n, _anorm);
}

template <class T> void
BiDiagonal<T>::initialize_rotation(size_t m, size_t n, T& x, T& y) const
{
    const T	g    = ((_diagonal[n]     + _diagonal[n-1])*
			(_diagonal[n]     - _diagonal[n-1])+
			(_off_diagonal[n] + _off_diagonal[n-1])*
			(_off_diagonal[n] - _off_diagonal[n-1]))
		     / (2.0*_diagonal[n-1]*_off_diagonal[n]),
      // Caution!! You have to ensure that _diagonal[n-1] != 0
      // as well as _off_diagonal[n].
		absg = std::fabs(g),
		gg1  = (absg > 1.0 ?
			absg * std::sqrt(1.0 + (1.0/absg)*(1.0/absg)) :
			std::sqrt(1.0 + absg*absg)),
		t    = (g > 0.0 ? g + gg1 : g - gg1);
    x = ((_diagonal[m] + _diagonal[n])*(_diagonal[m] - _diagonal[n]) -
	 _off_diagonal[n]*(_off_diagonal[n] + _diagonal[n-1]/t)) / _diagonal[m];
  //x = _diagonal[m];				// without shifting
    y = _off_diagonal[m+1];
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
  private:
    typedef BiDiagonal<T>			super;
    
  public:
    typedef T					element_type;	//!< 成分の型

  public:
  //! 与えられた一般行列の特異値分解を求める．
  /*!
    \param a	特異値分解する一般行列
  */
    template <class E,
	      typename std::enable_if<is_range<E>::value>::type* = nullptr>
    SVDecomposition(const E& a)
	:super(a)				{super::diagonalize();}

    using	super::nrow;
    using	super::ncol;
    using	super::Ut;
    using	super::Vt;
    using	super::diagonal;

  //! 特異値を求める．
  /*!
    \param i	絶対値の大きい順に並んだ特異値の1つを指定するindex
    \return	指定されたindexに対応する特異値
  */
    const T&	operator [](int i)	const	{return diagonal()[i];}
};

/************************************************************************
*  global functions							*
************************************************************************/
template <class T, class B> const Vector<T, B>&
serialize(const Vector<T, B>& x)
{
    return x;
}
template <class T, class B, class R> Vector<T>
serialize(const Matrix<T, B, R>& x)
{
    return Vector<T>(const_cast<T*>(x.data()), x.nrow() * x.ncol());
}
    
/************************************************************************
*  type aliases								*
************************************************************************/
template <class T, size_t D>
using FixedSizedVector = Vector<T, Buf<T, D> >;
template <class T, size_t R, size_t C>
using FixedSizedMatrix = Matrix<T, Buf<T, R*C>, Buf<Vector<T>, R> >;

typedef FixedSizedVector<short,  2>	Vector2s;
typedef FixedSizedVector<int,    2>	Vector2i;
typedef FixedSizedVector<float,  2>	Vector2f;
typedef FixedSizedVector<double, 2>	Vector2d;
typedef FixedSizedVector<short,  3>	Vector3s;
typedef FixedSizedVector<int,    3>	Vector3i;
typedef FixedSizedVector<float,  3>	Vector3f;
typedef FixedSizedVector<double, 3>	Vector3d;
typedef FixedSizedVector<short,  4>	Vector4s;
typedef FixedSizedVector<int,    4>	Vector4i;
typedef FixedSizedVector<float,  4>	Vector4f;
typedef FixedSizedVector<double, 4>	Vector4d;
typedef FixedSizedMatrix<float,  2, 2>	Matrix22f;
typedef FixedSizedMatrix<double, 2, 2>	Matrix22d;
typedef FixedSizedMatrix<float,  2, 3>	Matrix23f;
typedef FixedSizedMatrix<double, 2, 3>	Matrix23d;
typedef FixedSizedMatrix<float,  3, 3>	Matrix33f;
typedef FixedSizedMatrix<double, 3, 3>	Matrix33d;
typedef FixedSizedMatrix<float,  3, 4>	Matrix34f;
typedef FixedSizedMatrix<double, 3, 4>	Matrix34d;
typedef FixedSizedMatrix<float,  4, 4>	Matrix44f;
typedef FixedSizedMatrix<double, 4, 4>	Matrix44d;
    
}
#endif	// !__TU_VECTORPP_H






