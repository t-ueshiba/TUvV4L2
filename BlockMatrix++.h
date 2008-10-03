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
 *  $Id: BlockMatrix++.h,v 1.9 2008-10-03 04:23:37 ueshiba Exp $
 */
#ifndef __TUBlockMatrixPP_h
#define __TUBlockMatrixPP_h

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class BlockMatrix<T>							*
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
class BlockMatrix : public Array<Matrix<T> >
{
  public:
  //! 指定された個数の小行列から成るブロック対角行列を生成する．
  /*!
    \param d	小行列の個数
  */
    explicit BlockMatrix(u_int d=0)	:Array<Matrix<T> >(d)	{}
    BlockMatrix(const Array<u_int>& nrows,
		const Array<u_int>& ncols)			;

    using		Array<Matrix<T> >::dim;
    u_int		nrow()				const	;
    u_int		ncol()				const	;
    BlockMatrix		trns()				const	;
    BlockMatrix&	operator  =(const T& c)			;
    BlockMatrix&	operator *=(double c)
			{Array<Matrix<T> >::operator *=(c); return *this;}
    BlockMatrix&	operator /=(double c)
			{Array<Matrix<T> >::operator /=(c); return *this;}

  //! このブロック対角行列に他のブロック対角行列を足す．
  /*!
    \param b	足すブロック対角行列
    \return	このブロック対角行列
  */
    BlockMatrix&	operator +=(const BlockMatrix& b)
			{Array<Matrix<T> >::operator +=(b); return *this;}

  //! このブロック対角行列から他のブロック対角行列を引く．
  /*!
    \param b	引くブロック対角行列
    \return	このブロック対角行列
  */
    BlockMatrix&	operator -=(const BlockMatrix& b)
			{Array<Matrix<T> >::operator -=(b); return *this;}

			operator Matrix<T>()		const	;
};

/************************************************************************
*  numeric operators							*
************************************************************************/
template <class T> BlockMatrix<T>
operator *(const BlockMatrix<T>& a, const BlockMatrix<T>& b)	;

template <class T> Matrix<T>
operator *(const BlockMatrix<T>& b, const Matrix<T>& m)		;

template <class T> Matrix<T>
operator *(const Matrix<T>& m, const BlockMatrix<T>& b)		;

template <class T> Vector<T>
operator *(const BlockMatrix<T>& b, const Vector<T>& v)		;

template <class T> Vector<T>
operator *(const Vector<T>& v, const BlockMatrix<T>& b)		;
 
}

#endif	/* !__TUBlockMatrixPP_h	*/
