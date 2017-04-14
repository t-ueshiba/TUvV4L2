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
    explicit BlockDiagonalMatrix(size_t d=0)	:super(d)	{}
    BlockDiagonalMatrix(const Array<size_t>& nrows,
			const Array<size_t>& ncols)		;

    using		super::size;
    size_t		nrow()				const	;
    size_t		ncol()				const	;
    BlockDiagonalMatrix	trns()				const	;
			operator Matrix<T>()		const	;
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
	(*this)[i].resize(nrows[i], ncols[i]);
}

//! ブロック対角行列の総行数を返す．
/*!
  \return	総行数
*/
template <class T> size_t
BlockDiagonalMatrix<T>::nrow() const
{
    size_t	r = 0;
    for (const auto& block : *this)
	r += block.nrow();
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
    for (const auto& block : *this)
	c += block.ncol();
    return c;
}

//! ブロック対角行列から通常の行列を生成する.
/*!
  \param m	ブロック対角行列
*/
template <class T>
BlockDiagonalMatrix<T>::operator Matrix<T>() const
{
    Matrix<T>	m(nrow(), ncol());
    size_t	r = 0, c = 0;
    for (const auto& block : *this)
    {
	m(r, block.nrow(), c, block.ncol()) = block;
	r += block.nrow();
	c += block.ncol();
    }

    
    return m;
}
    
/************************************************************************
*  numeric operators							*
************************************************************************/
//! ブロック対角行列の転置行列を返す．
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
transpose(const BlockDiagonalMatrix<T>& b)
{
    BlockDiagonalMatrix<T>	val(b.size());
    for (size_t i = 0; i < val.size(); ++i)
	val[i] = transpose(b[i]);
    return std::move(val);
}

//! 2つのブロック対角行列の積
/*!
  \param a	第1引数
  \param b	第2引数
  \return	結果のブロック対角行列
*/
template <class T> BlockDiagonalMatrix<T>
operator *(const BlockDiagonalMatrix<T>& a, const BlockDiagonalMatrix<T>& b)
{
    assert(a.size() == b.size());
    BlockDiagonalMatrix<T>	val(a.size());
    for (size_t i = 0; i < val.size(); ++i)
	val[i] = a[i] * b[i];
    return std::move(val);
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
	val(r, b[i].nrow(), 0, m.ncol())
	    = b[i] * m(c, b[i].ncol(), 0, m.ncol());
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (c != m.nrow())
	throw std::invalid_argument("TU::operator *(const BlockDiagonalMatrix<T>&, const Matrix<T>&): dimension mismatch!!");
    return std::move(val);
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
	val(0, m.nrow(), c, b[i].ncol())
	    = m(0, m.nrow(), r, b[i].nrow()) * b[i];
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (r != m.ncol())
	throw std::invalid_argument("TU::operator *(const Matrix<T>&, const BlockDiagonalMatrix<T>&): dimension mismatch!!");
    return std::move(val);
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
    return std::move(val);
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
    return std::move(val);
}
 
}
#endif	// !__TU_BLOCKDIAGONALMATRIXPP_H
