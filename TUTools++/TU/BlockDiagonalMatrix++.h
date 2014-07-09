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
  \file		BlockDiagonalMatrix++.h
  \brief	クラス TU::BlockDiagonalMatrix の定義と実装
*/
#ifndef __TU_BLOCKDIAGONALMATRIXPP_H
#define __TU_BLOCKDIAGONALMATRIXPP_H

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class BlockDiagonalMatrix<T>						*
************************************************************************/
//! T型の要素を持つ小行列から成るブロック対角行列を表すクラス
/*!
  具体的にはd個の小行列\f$\TUvec{B}{1}, \TUvec{B}{2},\ldots, \TUvec{B}{d}\f$
  (同一サイズとは限らない)から成る
  \f$
  \TUvec{B}{} =
  \TUbeginarray{cccc}
  \TUvec{B}{1} & & & \\ & \TUvec{B}{2} & & \\ & & \ddots & \\
  & & & \TUvec{B}{d}
  \TUendarray
  \f$
  なる形の行列．
  \param T	要素の型
*/
template <class T>
class BlockDiagonalMatrix : public Array<Matrix<T> >
{
  private:
    typedef Array<Matrix<T> >	super;
    
  public:
    typedef T			element_type;
    
  public:
  //! 指定された個数の小行列から成るブロック対角行列を生成する．
  /*!
    \param d	小行列の個数
  */
    explicit BlockDiagonalMatrix(size_t d=0)	:super(d)		{}
    BlockDiagonalMatrix(const Array<size_t>& nrows,
			const Array<size_t>& ncols)			;

    using			super::size;
    size_t			nrow()				const	;
    size_t			ncol()				const	;
    BlockDiagonalMatrix		trns()				const	;
    BlockDiagonalMatrix&	operator =(element_type c)		;

  //! このブロック対角行列の全ての成分に同一の数値を掛ける．
  /*!
    \param c	掛ける数値
    \return	このブロック対角行列
  */
    BlockDiagonalMatrix&	operator *=(element_type c)
				{
				    super::operator *=(c);
				    return *this;
				}

  //! このブロック対角行列の全ての成分を同一の数値で割る．
  /*!
    \param c	割る数値
    \return	このブロック対角行列
  */
    template <class T2>
    BlockDiagonalMatrix&	operator /=(T2 c)
				{
				    super::operator /=(c);
				    return *this;
				}

  //! このブロック対角行列に他のブロック対角行列を足す．
  /*!
    \param b	足すブロック対角行列
    \return	このブロック対角行列
  */
    BlockDiagonalMatrix&	operator +=(const BlockDiagonalMatrix& b)
				{
				    super::operator +=(b);
				    return *this;
				}

  //! このブロック対角行列から他のブロック対角行列を引く．
  /*!
    \param b	引くブロック対角行列
    \return	このブロック対角行列
  */
    BlockDiagonalMatrix&	operator -=(const BlockDiagonalMatrix& b)
				{
				    super::operator -=(b);
				    return *this;
				}
};

//! 各小行列のサイズを指定してブロック対角行列を生成し，全要素を0で初期化する．
/*!
  \param nrows	各小行列の行数を順に収めた配列
  \param ncols	各小行列の列数を順に収めた配列
*/
template <class T>
BlockDiagonalMatrix<T>::BlockDiagonalMatrix(const Array<size_t>& nrows,
					    const Array<size_t>& ncols)
    :Array<Matrix<T> >(nrows.size())
{
    if (nrows.size() != ncols.size())
	throw std::invalid_argument("TU::BlockDiagonalMatrix<T>::BlockDiagonalMatrix: dimension mismatch between nrows and ncols!!");
    for (size_t i = 0; i < size(); ++i)
    {
	(*this)[i].resize(nrows[i], ncols[i]);
	(*this)[i] = element_type(0);
    }
}

//! ブロック対角行列の総行数を返す．
/*!
  \return	総行数
*/
template <class T> size_t
BlockDiagonalMatrix<T>::nrow() const
{
    size_t	r = 0;
    for (size_t i = 0; i < size(); ++i)
	r += (*this)[i].nrow();
    return r;
}

//! ブロック対角行列の総列数を返す．
/*!
  \return	総列数
*/
template <class T> size_t
BlockDiagonalMatrix<T>::ncol() const
{
    size_t	c = 0;
    for (size_t i = 0; i < size(); ++i)
	c += (*this)[i].ncol();
    return c;
}

//! このブロック対角行列の転置行列を返す．
/*!
  \return	転置行列，すなわち
  \f$
  \TUtvec{B}{} =
  \TUbeginarray{cccc}
  \TUtvec{B}{1} & & & \\ & \TUtvec{B}{2} & & \\ & & \ddots & \\
  & & & \TUtvec{B}{d}
  \TUendarray
  \f$
*/
template <class T> BlockDiagonalMatrix<T>
BlockDiagonalMatrix<T>::trns() const
{
    BlockDiagonalMatrix	val(size());
    for (size_t i = 0; i < val.size(); ++i)
	val[i] = (*this)[i].trns();
    return val;
}

//! このブロック対角行列の全ての小行列の全要素に同一の数値を代入する．
/*!
  \param c	代入する数値
  \return	このブロック対角行列
*/
template <class T> BlockDiagonalMatrix<T>&
BlockDiagonalMatrix<T>::operator =(element_type c)
{
    for (size_t i = 0; i < size(); ++i)
	(*this)[i] = c;
    return *this;
}

//! ブロック対角行列から通常の行列を生成する.
/*!
  \param m	ブロック対角行列
*/
template <class T, class B, class R> inline
Matrix<T, B, R>::Matrix(const BlockDiagonalMatrix<T>& m)
    :super(m.nrow(), m.ncol())
{
    size_t	r = 0, c = 0;
    for (size_t i = 0; i < m.size(); ++i)
    {
	(*this)(r, c, m[i].nrow(), m[i].ncol()) = m[i];
	r += m[i].nrow();
	c += m[i].ncol();
    }
}

//! ブロック対角行列から通常の行列に代入する.
/*!
  \param m	ブロック対角行列
  \return	この行列
*/
template <class T, class B, class R> inline Matrix<T, B, R>&
Matrix<T, B, R>::operator =(const BlockDiagonalMatrix<T>& m)
{
    super::resize(m.nrow(), m.ncol());
    size_t	r = 0, c = 0;
    for (size_t i = 0; i < m.size(); ++i)
    {
	(*this)(r, c, m[i].nrow(), m[i].ncol()) = m[i];
	r += m[i].nrow();
	c += m[i].ncol();
    }
}
    
/************************************************************************
*  numeric operators							*
************************************************************************/
//! 2つのブロック対角行列の積
/*!
  \param a	第1引数
  \param b	第2引数
  \return	結果のブロック対角行列
*/
template <class T> BlockDiagonalMatrix<T>
operator *(const BlockDiagonalMatrix<T>& a, const BlockDiagonalMatrix<T>& b)
{
    a.check_size(b.size());
    BlockDiagonalMatrix<T>	val(a.size());
    for (size_t i = 0; i < val.size(); ++i)
	val[i] = a[i] * b[i];
    return val;
}

//! ブロック対角行列と通常の行列の積
/*!
  \param b	第1引数(ブロック対角行列)
  \param m	第2引数(通常行列)
  \return	結果の通常行列
*/
template <class T> Matrix<T>
operator *(const BlockDiagonalMatrix<T>& b, const Matrix<T>& m)
{
    Matrix<T>	val(b.nrow(), m.ncol());
    size_t	r = 0, c = 0;
    for (size_t i = 0; i < b.size(); ++i)
    {
	val(r, 0, b[i].nrow(), m.ncol())
	    = b[i] * m(c, 0, b[i].ncol(), m.ncol());
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (c != m.nrow())
	throw std::invalid_argument("TU::operator *(const BlockDiagonalMatrix<T>&, const Matrix<T>&): dimension mismatch!!");
    return val;
}

//! 通常の行列とブロック対角行列の積
/*!
  \param m	第1引数(通常行列)
  \param b	第2引数(ブロック対角行列)
  \return	結果の通常行列
*/
template <class T> Matrix<T>
operator *(const Matrix<T>& m, const BlockDiagonalMatrix<T>& b)
{
    Matrix<T>	val(m.nrow(), b.ncol());
    size_t	r = 0, c = 0;
    for (size_t i = 0; i < b.size(); ++i)
    {
	val(0, c, m.nrow(), b[i].ncol())
	    = m(0, r, m.nrow(), b[i].nrow()) * b[i];
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (r != m.ncol())
	throw std::invalid_argument("TU::operator *(const Matrix<T>&, const BlockDiagonalMatrix<T>&): dimension mismatch!!");
    return val;
}

//! ブロック対角行列とベクトルの積
/*!
  \param b	ブロック対角行列
  \param v	ベクトル
  \return	結果のベクトル
*/
template <class T> Vector<T>
operator *(const BlockDiagonalMatrix<T>& b, const Vector<T>& v)
{
    Vector<T>	val(b.nrow());
    size_t	r = 0, c = 0;
    for (size_t i = 0; i < b.size(); ++i)
    {
	val(r, b[i].nrow()) = b[i] * v(c, b[i].ncol());
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (c != v.size())
	throw std::invalid_argument("TU::operator *(const BlockDiagonalMatrix<T>&, const Vector<T>&): dimension mismatch!!");
    return val;
}

//! ベクトルとブロック対角行列の積
/*!
  \param v	ベクトル
  \param b	ブロック対角行列
  \return	結果のベクトル
*/
template <class T> Vector<T>
operator *(const Vector<T>& v, const BlockDiagonalMatrix<T>& b)
{
    Vector<T>	val(b.ncol());
    size_t	r = 0, c = 0;
    for (size_t i = 0; i < b.size(); ++i)
    {
	val(c, b[i].ncol()) = v(r, b[i].nrow()) * b[i];
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (r != v.size())
	throw std::invalid_argument("TU::operator *(const Vector<T>&, const BlockDiagonalMatrix<T>&): dimension mismatch!!");
    return val;
}
 
}
#endif	// !__TU_BLOCKDIAGONALMATRIXPP_H
