/*
 *  平成21-22年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2009-2010.
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
 *  $Id: BandMatrix++.h 1235 2012-11-29 12:30:58Z ueshiba $
 */
/*!
  \file		BandMatrix++.h
  \brief	クラス TU::BandMatrix の定義と実装
*/
#ifndef __TU_BANDMATRIXPP_H
#define __TU_BANDMATRIXPP_H

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class BandMatrix<T, P, Q>						*
************************************************************************/
//! 帯行列を表すクラス
/*!
  \param T	成分の型
  \param P	左帯幅(j < i-P なるjに対して(i, j)成分が0になるような非負整数)
  \param Q	右帯幅(j > i+P なるjに対して(i, j)成分が0になるような非負整数)
*/
template <class T, size_t P, size_t Q>
class BandMatrix
{
  public:
    typedef T				element_type;	//!< 成分の型
    typedef Matrix<element_type>	matrix_type;	//!< 同型の成分を持つ行列
    
  private:
    typedef Vector<T, Buf<T, P+Q+1> >	RowData;
    
  public:
  // 構造操作
    explicit BandMatrix(size_t siz=P+Q+1)				;
    void		resize(size_t siz)				;

  // 基本情報
    static size_t	leftBandWidth()					;
    static size_t	rightBandWidth()				;
    size_t		size()					const	;
    element_type	operator ()(size_t i, size_t j)		const	;
    element_type&	operator ()(size_t i, size_t j)			;
    matrix_type		A()					const	;
    matrix_type		L()					const	;
    matrix_type		U()					const	;

  // 演算
    BandMatrix&		operator =(element_type c)			;
    BandMatrix&		decompose()					;
    template <class T2, class B2>
    void		substitute(Vector<T2, B2>& b)		const	;
    
  // 出力
    std::ostream&	put(std::ostream& out)			const	;
    
  private:
    size_t		colBegin(size_t i)			const	;
    size_t		colEnd(size_t i)			const	;
    size_t		rowBegin(size_t j)			const	;
    size_t		rowEnd(size_t j)			const	;
    
  private:
    Array<RowData>	_buf;
};

//! 帯行列を生成する.
/*!
  \param siz	帯行列の行と列のサイズ(次元)
*/
template <class T, size_t P, size_t Q>
BandMatrix<T, P, Q>::BandMatrix(size_t siz)
    :_buf(siz)
{
    if (siz < P+1 || siz < Q+1)
	throw std::invalid_argument("TU::BandMatrix<T, P, Q>::BandMatrix(): too small dimension!");
}
    
//! 帯行列のサイズを変更し，すべての成分を0にする.
/*!
  \param siz	帯行列の行と列のサイズ(次元)
*/
template <class T, size_t P, size_t Q> inline void
BandMatrix<T, P, Q>::resize(size_t siz)
{
    if (siz < P+1 || siz < Q+1)
	throw std::invalid_argument("TU::BandMatrix<T, P, Q>::BandMatrix(): too small dimension!");

    _buf.resize(siz);
    *this = element_type(0);
}
    
//! 帯行列の左帯幅を返す.
/*!
  \return	左帯幅
*/
template <class T, size_t P, size_t Q> inline size_t
BandMatrix<T, P, Q>::leftBandWidth()
{
    return P;
}
    
//! 帯行列の右帯幅を返す.
/*!
  \return	右帯幅
*/
template <class T, size_t P, size_t Q> inline size_t
BandMatrix<T, P, Q>::rightBandWidth()
{
    return Q;
}
    
//! 帯行列の行と列のサイズ(次元)を返す.
/*!
  \return	行と列のサイズ(次元)
*/
template <class T, size_t P, size_t Q> inline size_t
BandMatrix<T, P, Q>::size() const
{
    return _buf.size();
}
    
//! 帯行列の指定された成分への定数参照を返す.
/*!
  与えるindexは非零成分を指すものでなければならない.
  \param i	行を指定するindex
  \param j	列を指定するindex
  \return	(i, j)成分への定数参照
*/
template <class T, size_t P, size_t Q>
inline typename BandMatrix<T, P, Q>::element_type
BandMatrix<T, P, Q>::operator ()(size_t i, size_t j) const
{
    return _buf[i][P+j-i];
}
    
//! 帯行列の指定された成分への参照を返す.
/*!
  与えるindexは非零成分を指すものでなければならない.
  \param i	行を指定するindex
  \param j	列を指定するindex
  \return	(i, j)成分への参照
*/
template <class T, size_t P, size_t Q>
inline typename BandMatrix<T, P, Q>::element_type&
BandMatrix<T, P, Q>::operator ()(size_t i, size_t j)
{
    return _buf[i][P+j-i];
}
    
//! 帯行列を通常の密行列の形式に変換する.
/*!
  \return	変換された密行列
*/
template <class T, size_t P, size_t Q>
inline typename BandMatrix<T, P, Q>::matrix_type
BandMatrix<T, P, Q>::A() const
{
    matrix_type	m(size(), size());

    for (size_t i = 0; i < size(); ++i)
    {
	const size_t	je = colEnd(i);
	for (size_t j = colBegin(i); j < je; ++j)
	    m[i][j] = (*this)(i, j);
    }
    
    return m;
}
    
//! 帯行列の下半三角部分を通常の密行列の形式で返す.
/*!
  左帯幅は leftBandWidth() に等しい.
  \return	下半三角部分
*/
template <class T, size_t P, size_t Q>
inline typename BandMatrix<T, P, Q>::matrix_type
BandMatrix<T, P, Q>::L() const
{
    matrix_type	m(size(), size());

    for (size_t i = 0; i < size(); ++i)
	for (size_t j = colBegin(i); j <= i; ++j)
	    m[i][j] = (*this)(i, j);
    
    return m;
}
    
//! 帯行列の上半三角部分を通常の密行列の形式で返す.
/*!
  右帯幅は rightBandWidth() に等しく，対角成分はすべて1である.
  \return	上半三角部分
*/
template <class T, size_t P, size_t Q>
inline typename BandMatrix<T, P, Q>::matrix_type
BandMatrix<T, P, Q>::U() const
{
    matrix_type	m(size(), size());

    for (size_t i = 0; i < size(); ++i)
    {
	m[i][i] = 1;
	const size_t	je = colEnd(i);
	for (size_t j = i + 1; j < je; ++j)
	    m[i][j] = (*this)(i, j);
    }
    
    return m;
}
    
//! 帯行列のすべての成分に同一の値を代入する.
/*!
  \param c	代入する値
  \return	この帯行列
*/
template <class T, size_t P, size_t Q> inline BandMatrix<T, P, Q>&
BandMatrix<T, P, Q>::operator =(element_type c)
{
    _buf = c;
}

//! 帯行列を下半三角行列と上半三角行列の積に分解する(LU分解).
/*!
  下半三角行列の左帯幅は leftBandWidth() に等しい.
  上半三角行列の右帯幅は rightBandWidth() に等しく，対角成分はすべて1である.
  \return	分解されたこの帯行列
*/
template <class T, size_t P, size_t Q> BandMatrix<T, P, Q>&
BandMatrix<T, P, Q>::decompose()
{
    for (size_t n = 0; n < size(); ++n)
    {
	const element_type	a = (*this)(n, n);
	if (a == element_type(0))
	    throw std::runtime_error("TU::BandMatrix<T, P, Q>::decompose(): sigular matrix!");
	
	const size_t	je = colEnd(n);
	for (size_t j = n + 1; j < je; ++j)
	    (*this)(n, j) /= a;

	const size_t	ie = rowEnd(n);
	for (size_t i = n + 1; i < ie; ++i)
	{
	    const element_type	b  = (*this)(i, n);
	    for (size_t j = n + 1; j < je; ++j)
		(*this)(i, j) -= b * (*this)(n, j);
	}
    }
    
    return *this;
}

//! もとの帯行列を係数行列とした連立1次方程式を解く．
/*!
  \param b			もとの帯行列\f$\TUvec{A}{}\f$と同じ次
				元を持つベクトル．\f$\TUtvec{b}{} =
				\TUtvec{x}{}\TUvec{A}{}\f$の解に変換さ
				れる．
  \throw std::invalid_argument	ベクトルbの次元がもとの正方行列の次元に一致
				しない場合に送出
  \throw std::runtime_error	もとの正方行列が正則でない場合に送出
*/
template <class T, size_t P, size_t Q> template <class T2, class B2> void
BandMatrix<T, P, Q>::substitute(Vector<T2, B2>& b) const
{
    if (b.size() != size())
	throw std::invalid_argument("TU::BandMatrix<T, P, Q>::substitute(): Dimension of given vector is not equalt to mine!");

    for (size_t j = 0; j < b.size(); ++j)
	for (size_t i = rowBegin(j); i < j; ++i)
	    b[j] -= b[i] * (*this)(i, j);	// forward substitution

    for (size_t j = b.size(); j-- > 0; )
    {
	for (size_t i = rowEnd(j); --i > j; )
	    b[j] -= b[i] * (*this)(i, j);	// backward substitution
	if ((*this)(j, j) == element_type(0))
	    throw std::runtime_error("TU::BandMatrix<T, P, Q>::substitute(): sigular matrix!");
	b[j] /= (*this)(j, j);
    }
}

//! 出力ストリームに帯行列を書き出す(ASCII).
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T, size_t P, size_t Q> std::ostream&
BandMatrix<T, P, Q>::put(std::ostream& out) const
{
    for (size_t i = 0; i < size(); ++i)
    {
	const size_t	jb = colBegin(i), je = colEnd(i);
	for (size_t j = 0; j < jb; ++j)
	    out << " _";
	for (size_t j = jb; j < je; ++j)
	    out << ' ' << (*this)(i, j);
	for (size_t j = je; j < size(); ++j)
	    out << " _";
	out << std::endl;
    }

    return out;
}

template <class T, size_t P, size_t Q> inline size_t
BandMatrix<T, P, Q>::colBegin(size_t i) const
{
    return (i > P ? i - P : 0);
}

template <class T, size_t P, size_t Q> inline size_t
BandMatrix<T, P, Q>::colEnd(size_t i) const
{
    return (i + Q < size() ? i + Q + 1 : size());
}

template <class T, size_t P, size_t Q> inline size_t
BandMatrix<T, P, Q>::rowBegin(size_t j) const
{
    return (j > Q ? j - Q : 0);
}

template <class T, size_t P, size_t Q> inline size_t
BandMatrix<T, P, Q>::rowEnd(size_t j) const
{
    return (j + P < size() ? j + P + 1 : size());
}

/************************************************************************
*  global functions							*
************************************************************************/
//! 出力ストリームに帯行列を書き出す(ASCII).
/*!
  帯行列の全ての成分が出力されたのち，最後に空行が出力される.
  \param out	出力ストリーム
  \param A	帯行列
  \return	outで指定した出力ストリーム
*/
template <class T, size_t P, size_t Q> inline std::ostream&
operator <<(std::ostream& out, const BandMatrix<T, P, Q>& A)
{
    return A.put(out) << std::endl;
}

}
#endif	// !__TU_BANDMATRIXPP_H
