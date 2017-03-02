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
 *  $Id: Vector++.h 1952 2016-03-05 05:37:36Z ueshiba $
 */
/*!
  \file		Vector++.h
  \brief	ベクトルと行列およびそれに関連するクラスの定義と実装
*/
#ifndef __TU_VECTORPP_H
#define __TU_VECTORPP_H

#include "TU/Array++.h"
#include <cmath>

namespace TU
{
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
    const size_t	_p, _q;		// rotation axis
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

//! この行列の左から（転置された）回転行列を掛ける．
/*!
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow\TUtvec{R}{}\TUvec{A}{}\f$
*/
template <class T, size_t R, size_t C> Array2<T, R, C>&
rotate_from_left(Array2<T, R, C>& m, const Rotation<T>& r)
{
    for (size_t j = 0; j < m.ncol(); ++j)
    {
	const auto	tmp = m[r.p()][j];
	
	m[r.p()][j] =  r.cos()*tmp + r.sin()*m[r.q()][j];
	m[r.q()][j] = -r.sin()*tmp + r.cos()*m[r.q()][j];
    }
    return m;
}

//! この行列の右から回転行列を掛ける．
/*!
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow\TUvec{A}{}\TUvec{R}{}\f$
*/
template <class T, size_t R, size_t C> Array2<T, R, C>&
rotate_from_right(Array2<T, R, C>& m, const Rotation<T>& r)
{
    for (size_t i = 0; i < m.nrow(); ++i)
    {
	const auto	tmp = m[i][r.p()];
	
	m[i][r.p()] =  tmp*r.cos() + m[i][r.q()]*r.sin();
	m[i][r.q()] = -tmp*r.sin() + m[i][r.q()]*r.cos();
    }
    return m;
}
    
/************************************************************************
*  class LUDecomposition<T>						*
************************************************************************/
//! 正方行列のLU分解を表すクラス
template <class T, size_t N=0>
class LUDecomposition
{
  public:
    template <class E,
	      typename std::enable_if<rank<E>() == 2>::type* = nullptr>
    LUDecomposition(const E& m)				;

    template <class T2, size_t N2>
    void	substitute(Array<T2, N2>& b)	const	;

  //! もとの正方行列の行列式を返す．
  /*!
    \return	もとの正方行列の行列式
  */
    T		det()				const	{return _det;}
    
  private:
    Array2<T, N, N>	_m;
    Array<size_t, N>	_index;
    T			_det;
};

//! 与えられた正方行列のLU分解を生成する．
/*!
 \param m			LU分解する正方行列
 \throw std::invalid_argument	mが正方行列でない場合に送出
*/
template <class T>
template <class E, typename std::enable_if<is_range<E>::value>::type*>
LUDecomposition<T>::LUDecomposition(const E& m)
    :_m(m), _index(_m.ncol()), _det(1)
{
    if (_m.nrow() != _m.ncol())
        throw std::invalid_argument("TU::LUDecomposition<T>::LUDecomposition: not square matrix!!");

    for (size_t j = 0; j < _index.size(); ++j)	// initialize column index
	_index[j] = j;				// for explicit pivotting

    Array<T, N>	scale(_m.ncol());
    for (size_t j = 0; j < ncol(); ++j)	// find maximum abs. value in each col.
    {					// for implicit pivotting
	T max = 0;

	for (size_t i = 0; i < nrow(); ++i)
	{
	    const T tmp = std::fabs(m[i][j]);
	    if (tmp > max)
		max = tmp;
	}
	scale[j] = (max != 0 ? 1.0 / max : 1.0);
    }

    for (size_t i = 0; i < nrow(); ++i)
    {
	for (size_t j = 0; j < i; ++j)		// left part (j < i)
	{
	    T& sum = _m[i][j];
	    for (size_t k = 0; k < j; ++k)
		sum -= _m[i][k] * _m[k][j];
	}

	size_t	jmax = i;
	T	max = 0.0;
	for (size_t j = i; j < ncol(); ++j)  // diagonal and right part (i <= j)
	{
	    T& sum = _m[i][j];
	    for (size_t k = 0; k < i; ++k)
		sum -= _m[i][k] * _m[k][j];
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
		swap(_m[k][i], _m[k][jmax]);
	    std::swap(_index[i], _index[jmax]);	// swap column index
	    std::swap(scale[i], scale[jmax]);	// swap colum-wise scale factor
	    _det = -_det;
	}

	_det *= _m[i][i];

	if (_m[i][i] == 0)	// singular matrix ?
	    break;

	for (size_t j = i + 1; j < nrow(); ++j)
	    _m[i][j] /= _m[i][i];
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
template <class T> template <class T2, size_t D2> void
LUDecomposition<T>::substitute(Vector<T2, D2>& b) const
{
    if (b.size() != ncol())
	throw std::invalid_argument("TU::LUDecomposition<T>::substitute: Dimension of given vector is not equal to mine!!");
    
    Vector<T2, D2>	tmp(b);
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
template <class T, size_t D> template <class E>
inline typename std::enable_if<is_range<E>::value, Vector<T, D>&>::type
Vector<T, D>::solve(const E& m)
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
template <class T, size_t R, size_t C> template <class E>
typename std::enable_if<is_range<E>::value, Matrix<T, R, C>&>::type
Matrix<T, R, C>::solve(const E& m)
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
template <class T, size_t R, size_t C> inline T
Matrix<T, R, C>::det() const
{
    return LUDecomposition<T>(*this).det();
}

}
#endif	// !__TU_VECTORPP_H






