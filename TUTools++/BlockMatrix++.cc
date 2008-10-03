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
 *  $Id: BlockMatrix++.cc,v 1.9 2008-10-03 04:23:37 ueshiba Exp $
 */
#include "TU/BlockMatrix++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class BlockMatrix<T>							*
************************************************************************/
//! 各小行列のサイズを指定してブロック対角行列を生成し，全要素を0で初期化する．
/*!
  \param nrows	各小行列の行数を順に収めた配列
  \param ncols	各小行列の列数を順に収めた配列
*/
template <class T>
BlockMatrix<T>::BlockMatrix(const Array<u_int>& nrows,
			    const Array<u_int>& ncols)
    :Array<Matrix<T> >(nrows.dim())
{
    if (nrows.dim() != ncols.dim())
	throw std::invalid_argument("TU::BlockMatrix<T>::BlockMatrix: dimension mismatch between nrows and ncols!!");
    for (int i = 0; i < dim(); ++i)
    {
	(*this)[i].resize(nrows[i], ncols[i]);
	(*this)[i] = 0.0;
    }
}

//! ブロック対角行列の総行数を返す．
/*!
  \return	総行数
*/
template <class T> u_int
BlockMatrix<T>::nrow() const
{
    size_t	r = 0;
    for (int i = 0; i < dim(); ++i)
	r += (*this)[i].nrow();
    return r;
}

//! ブロック対角行列の総列数を返す．
/*!
  \return	総列数
*/
template <class T> u_int
BlockMatrix<T>::ncol() const
{
    size_t	c = 0;
    for (int i = 0; i < dim(); ++i)
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
template <class T> BlockMatrix<T>
BlockMatrix<T>::trns() const
{
    BlockMatrix	val(dim());
    for (int i = 0; i < val.dim(); ++i)
	val[i] = (*this)[i].trns();
    return val;
}

//! このブロック対角行列の全ての小行列の全要素に同一の数値を代入する．
/*!
  \param c	代入する数値
  \return	このブロック対角行列
*/
template <class T> BlockMatrix<T>&
BlockMatrix<T>::operator =(const T& c)
{
    for (int i = 0; i < dim(); ++i)
	(*this)[i] = c;
    return *this;
}

//! このブロック対角行列を通常の行列に変換する．
/*!
  \return	変換された行列
*/
template <class T>
BlockMatrix<T>::operator Matrix<T>() const
{
    Matrix<T>	val(nrow(), ncol());
    for (int r = 0, c = 0, i = 0; i < dim(); ++i)
    {
	val(r, c, (*this)[i].nrow(), (*this)[i].ncol()) = (*this)[i];
	r += (*this)[i].nrow();
	c += (*this)[i].ncol();
    }
    return val;
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
template <class T> BlockMatrix<T>
operator *(const BlockMatrix<T>& a, const BlockMatrix<T>& b)
{
    a.check_dim(b.dim());
    BlockMatrix<T>	val(a.dim());
    for (int i = 0; i < val.dim(); ++i)
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
operator *(const BlockMatrix<T>& b, const Matrix<T>& m)
{
    Matrix<T>	val(b.nrow(), m.ncol());
    int		r = 0, c = 0;
    for (int i = 0; i < b.dim(); ++i)
    {
	val(r, 0, b[i].nrow(), m.ncol())
	    = b[i] * m(c, 0, b[i].ncol(), m.ncol());
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (c != m.nrow())
	throw std::invalid_argument("TU::operaotr *(const BlockMatrix<T>&, const Matrix<T>&): dimension mismatch!!");
    return val;
}

//! 通常の行列とブロック対角行列の積
/*!
  \param m	第1引数(通常行列)
  \param b	第2引数(ブロック対角行列)
  \return	結果の通常行列
*/
template <class T> Matrix<T>
operator *(const Matrix<T>& m, const BlockMatrix<T>& b)
{
    Matrix<T>	val(m.nrow(), b.ncol());
    int		r = 0, c = 0;
    for (int i = 0; i < b.dim(); ++i)
    {
	val(0, c, m.nrow(), b[i].ncol())
	    = m(0, r, m.nrow(), b[i].nrow()) * b[i];
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (r != m.ncol())
	throw std::invalid_argument("TU::operaotr *(const Matrix<T>&, const BlockMatrix<T>&): dimension mismatch!!");
    return val;
}

//! ブロック対角行列とベクトルの積
/*!
  \param b	ブロック対角行列
  \param v	ベクトル
  \return	結果のベクトル
*/
template <class T> Vector<T>
operator *(const BlockMatrix<T>& b, const Vector<T>& v)
{
    Vector<T>	val(b.nrow());
    int		r = 0, c = 0;
    for (int i = 0; i < b.dim(); ++i)
    {
	val(r, b[i].nrow()) = b[i] * v(c, b[i].ncol());
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (c != v.dim())
	throw std::invalid_argument("TU::operaotr *(const BlockMatrix<T>&, const Vector<T>&): dimension mismatch!!");
    return val;
}

//! ベクトルとブロック対角行列の積
/*!
  \param v	ベクトル
  \param b	ブロック対角行列
  \return	結果のベクトル
*/
template <class T> Vector<T>
operator *(const Vector<T>& v, const BlockMatrix<T>& b)
{
    Vector<T>	val(b.ncol());
    int		r = 0, c = 0;
    for (int i = 0; i < b.dim(); ++i)
    {
	val(c, b[i].ncol()) = v(r, b[i].nrow()) * b[i];
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (r != v.dim())
	throw std::invalid_argument("TU::operaotr *(const Vector<T>&, const BlockMatrix<T>&): dimension mismatch!!");
    return val;
}
 
}
