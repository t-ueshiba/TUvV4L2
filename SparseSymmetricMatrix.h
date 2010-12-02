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
 *  $Id: SparseSymmetricMatrix.h,v 1.1 2010-12-02 07:26:36 ueshiba Exp $
 */
/*!
  \file		SparseMatrix.h
  \brief	クラス#TU::SparseSymmetricMatrixの定義と実装
*/
#ifndef __SPARSESYMMETRICMATRIX_H
#define __SPARSESYMMETRICMATRIX_H

#include "TU/Vector++.h"
#include <algorithm>
#include <mkl_pardiso.h>

namespace TU
{
/************************************************************************
*  class SparseSymmetricMatrix<T>					*
************************************************************************/
//! Intel Math-Kernel Library(MKL)のフォーマットによる疎対称行列
template <class T>
class SparseSymmetricMatrix
{
  public:
    typedef T	value_type;		//!< 成分の型
    
  public:
    SparseSymmetricMatrix(u_int d, u_int nelms)				;

    SparseSymmetricMatrix<T>&
		operator =(value_type c)				;
    u_int	dim()						const	;
    u_int	nelements()					const	;
    u_int	nelements(u_int i)				const	;
    T&		operator ()(u_int i)					;
    const T&	operator ()(u_int i)				const	;
    T&		operator ()(u_int i, u_int j)				;
    const T&	operator ()(u_int i, u_int j)			const	;
    template <class F, class S, class B>
    SparseSymmetricMatrix<T>&
		apply(u_int i, u_int j, F f, const Vector<S, B>& v)	;
    template <class F, class S, class B, class R>
    SparseSymmetricMatrix<T>&
		apply(u_int i, u_int j, F f, const Matrix<S, B, R>& M)	;
    Vector<T>	solve(const Vector<T>& b)			const	;

    template <class S> friend std::ostream&
		operator <<(std::ostream& out,
			    const SparseSymmetricMatrix<S>& A)		;
    template <class S> friend std::istream&
		operator >>(std::istream& in,
			    SparseSymmetricMatrix<S>& A)		;
    
  protected:
    u_int	index(u_int i, u_int j)				const	;
    
  private:
    static int	pardiso_precision()					;

  protected:
    Array<u_int>	_rowIndex;	//!< 各行の先頭成分の通し番号(Fortran形式)
    Array<u_int>	_columns;	//!< 各成分の列番号(Fortran形式)
    Vector<T>		_values;	//!< 各成分の値
};

//! 疎対称行列を生成する．
/*!
  \param d	行列の次元
  \param nelms	行列の成分数
*/
template <class T> inline
SparseSymmetricMatrix<T>::SparseSymmetricMatrix(u_int d, u_int nelms)
    :_rowIndex(d + 1), _columns(nelms), _values(_columns.dim())
{
    _rowIndex = 0;
    _rowIndex[dim()] = nelements() + 1;
    _columns = 0;
}
    
//! すべての0でない成分に定数を代入する．
/*!
  \param c	代入する定数
  \return	この疎対称行列
*/
template <class T> inline SparseSymmetricMatrix<T>&
SparseSymmetricMatrix<T>::operator =(value_type c)
{
    _values = c;

    return *this;
}
    
//! 行列の次元を返す．
/*!
  \return	行列の次元
*/
template <class T> inline u_int
SparseSymmetricMatrix<T>::dim() const
{
    return _rowIndex.dim() - 1;
}
    
//! 行列の全成分数を返す．
/*!
  \return	行列の全成分数
*/
template <class T> inline u_int
SparseSymmetricMatrix<T>::nelements() const
{
    return _columns.dim();
}
    
//! 指定された行の成分数を返す．
/*!
  \param i	行を指定するindex
  \return	第i行の成分数
*/
template <class T> inline u_int
SparseSymmetricMatrix<T>::nelements(u_int i) const
{
    return _rowIndex[i+1] - _rowIndex[i];
}
    
//! 指定された行の対角成分を返す．
/*!
  \param i	行を指定するindex
  \return	(i, i)成分
*/
template <class T> inline T&
SparseSymmetricMatrix<T>::operator ()(u_int i)
{
    return _values[_rowIndex[i] - 1];
}
    
//! 指定された行の対角成分を返す．
/*!
  \param i	行を指定するindex
  \return	(i, i)成分
*/
template <class T> inline const T&
SparseSymmetricMatrix<T>::operator ()(u_int i) const
{
    return _values[_rowIndex[i] - 1];
}
    
//! 指定された行と列に対応する成分を返す．
/*!
  \param i	行を指定するindex
  \param j	列を指定するindex
  \return	(i, j)成分
*/
template <class T> inline T&
SparseSymmetricMatrix<T>::operator ()(u_int i, u_int j)
{
    return _values[index(i, j) - 1];
}
    
//! 指定された行と列に対応する成分を返す．
/*!
  \param i	行を指定するindex
  \param j	列を指定するindex
  \return	(i, j)成分
*/
template <class T> inline const T&
SparseSymmetricMatrix<T>::operator ()(u_int i, u_int j) const
{
    return _values[index(i, j) - 1];
}

//! 与えられたベクトルの各成分を指定された成分を起点として行方向に適用する．
/*!
  対称行列であるから，i > jの場合は(j, i)成分を起点として列方向に適用することに同等である．
  \param i	行を指定するindex
  \param j	列を指定するindex
  \param f	T型，S型の引数をとりT型の値を返す2項演算子
  \param v	その各成分がfの第2引数となるベクトル
  \return	この疎対称行列
*/
template <class T> template <class F, class S, class B>
SparseSymmetricMatrix<T>&
SparseSymmetricMatrix<T>::apply(u_int i, u_int j, F f, const Vector<S, B>& v)
{
    if (i <= j)
    {
	T*	p = &(*this)(i, j);

	for (u_int dj = 0; dj < v.dim(); ++dj)
	{
	    *p = f(*p, v[dj]);
	    ++p;
	}
    }
    else
    {
	T*	p = &(*this)(j, i);

	for (u_int dj = 0; dj < v.dim(); ++dj)
	{
	    *p = f(*p, v[dj]);
	    p += (nelements(j + dj) - 1);
	}
    }
    
    return *this;
}
    
//! 与えられた行列の各成分を指定された成分を起点として行優先順に適用する．
/*!
  対称行列であるから，i > jの場合は(j, i)成分を起点として列優先順に適用することに同等である．
  \param i	行を指定するindex
  \param j	列を指定するindex
  \param f	T型，S型の引数をとりT型の値を返す2項演算子
  \param M	その各成分がfの第2引数となる行列
  \return	この疎対称行列
*/
template <class T> template <class F, class S, class B, class R>
SparseSymmetricMatrix<T>&
SparseSymmetricMatrix<T>::apply(u_int i, u_int j, F f, const Matrix<S, B, R>& M)
{
    if (i == j)
    {
	for (u_int di = 0; di < M.nrow(); ++di)
	{
	    T*	p = &(*this)(i + di);
				    
	    for (u_int dj = di; dj < M.ncol(); ++dj)
	    {
		*p = f(*p, M[di][dj]);
		++p;
	    }
	}
    }
    else if (i < j)
    {
	T*	p = &(*this)(i, j);

	for (u_int di = 0; di < M.nrow(); ++di)
	{
	    for (u_int dj = 0; dj < M.ncol(); ++dj)
	    {
		*p = f(*p, M[di][dj]);
		++p;
	    }

	    p += (nelements(i + di) - M.ncol() - 1);
	}
    }
    else
    {
	T*	p = &(*this)(j, i);

	for (u_int dj = 0; dj < M.ncol(); ++dj)
	{
	    for (u_int di = 0; di < M.nrow(); ++di)
	    {
		*p = f(*p, M[di][dj]);
		++p;
	    }

	    p += (nelements(j + dj) - M.nrow() - 1);
	}
    }
    
    return *this;
}
    
//! この行列を係数とする連立一次方程式を解く．
/*!
  MKL direct sparse solverによって
  \f$\TUvec{A}{}\TUvec{x}{} = \TUvec{b}{}\f$を解く．
  \param b	ベクトル
  \return	解ベクトル
*/
template <class T> Vector<T>
SparseSymmetricMatrix<T>::solve(const Vector<T>& b) const
{
    using namespace	std;

    if (b.dim() != dim())
	throw runtime_error("SparseSymmetricMatrix<T>::solve(): input vector with invalid dimension!");

  // pardiso の各種パラメータを設定する．
    _MKL_DSS_HANDLE_t	pt[64];		// pardisoの内部メモリへのポインタ
    for (int i = 0; i < 64; ++i)
	pt[i] = 0;
    _INTEGER_t		iparm[64];
    for (int i = 0; i < 64; ++i)
	iparm[i] = 0;
    iparm[0]  =  1;			// iparm[1-] にデフォルト値以外を指定
    iparm[1]  =  2;			// nested dissection algorithm
    iparm[9]  =  8;			// eps = 1.0e-8
    iparm[17] = -1;			// 分解の非零成分数をレポート
    iparm[20] =  1;			// Bunch and Kaufman pivoting
    iparm[27] = pardiso_precision();	// float または double を指定
    _INTEGER_t		maxfct = 1;	// その分解を保持するべき行列の数
    _INTEGER_t		mnum   = 1;	// 何番目の行列について解くかを指定
    _INTEGER_t		mtype  = -2;	// 実対称行列
    _INTEGER_t		phase  = 13;	// 行列の解析から反復による細密化まで
    _INTEGER_t		n      = dim();	// 連立一次方程式の式数
    _INTEGER_t		nrhs   = 1;	// Ax = b における b の列数
    _INTEGER_t		msglvl = 0;	// メッセージを出力しない
    _INTEGER_t		error  = 0;	// エラーコード
    Array<_INTEGER_t>	perm(n);	// permutationベクトル
    Vector<T>		x(n);		// 解ベクトル

  // 連立一次方程式を解く．
    PARDISO(&pt[0], &maxfct, &mnum, &mtype, &phase, &n, (void*)&_values[0],
	    (_INTEGER_t*)&_rowIndex[0], (_INTEGER_t*)&_columns[0],
	    &perm[0], &nrhs, iparm, &msglvl, (void*)&b[0], &x[0], &error);
    if (error != 0)
	throw runtime_error("SparseSymmetricMatrix<T>::solve(): PARDISO failed!");

  // pardiso 内で使用した全メモリを解放する．
    phase = -1;
    PARDISO(&pt[0], &maxfct, &mnum, &mtype, &phase, &n, (void*)&_values[0],
	    (_INTEGER_t*)&_rowIndex[0], (_INTEGER_t*)&_columns[0],
	    &perm[0], &nrhs, iparm, &msglvl, (void*)&b[0], &x[0], &error);
    if (error != 0)
	throw runtime_error("SparseSymmetricMatrix<T>::solve(): PARDISO failed to release memory!");

    return x;
}

//! 指定された行と列に対応する#_valuesのindexを返す．
/*!
  #_rowIndexと同様にFortran形式のindexを返すので，実際に#_valuesにアクセスする際には
  返された値から1を引く必要がある．
  \param i			行を指定するindex
  \param j			列を指定するindex
  \return			#_valuesのindex(Fortran形式)
  \throw std::out_of_range	この行列が(i, j)成分を持たない場合に送出
*/
template <class T> inline u_int
SparseSymmetricMatrix<T>::index(u_int i, u_int j) const
{
    if (i > j)
	std::swap(i, j);

    ++j;				// 列indexをFortran形式に直す

    u_int	n = _rowIndex[i];	// 第i行の成分を指すindex(Fortran形式)
    for (; n < _rowIndex[i+1]; ++n)
	if (_columns[n-1] == j)
	    return n;
    if (n == 0)
	throw std::out_of_range("TU::SparseSymmetricMatrix<T>::index(): non-existent element!");
    return 0;
}
    
template<> inline int
SparseSymmetricMatrix<float>::pardiso_precision()	{return 1;}
template<> inline int
SparseSymmetricMatrix<double>::pardiso_precision()	{return 0;}

//! 出力ストリームへ疎対称行列を書き出し(ASCII)，さらに改行コードを出力する．
/*!
  \param out	出力ストリーム
  \param a	書き出す疎対称行列
  \return	outで指定した出力ストリーム
*/
template <class T> std::ostream&
operator <<(std::ostream& out, const SparseSymmetricMatrix<T>& A)
{
    using namespace	std;

    u_int	n = 0;
    for (u_int i = 0; i < A.dim(); ++i)
    {
	for (u_int j = 0; j < i; ++j)
	    out << " *";

	for (u_int j = i; j < A.dim(); ++j)
	{
	    if ((n < A._rowIndex[i+1] - 1) && (j == A._columns[n] - 1))
		out << ' ' << A._values[n++];
	    else
		out << " (0)";
	}
	out << endl;
    }

    return out << endl;
}

}

#endif	// ! __SPARSESYMMETRICMATRIX_H
