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
 *  $Id: SparseMatrix++.h,v 1.4 2011-09-14 04:41:10 ueshiba Exp $
 */
/*!
  \file		SparseMatrix++.h
  \brief	クラス TU::SparseMatrix の定義と実装
*/
#ifndef __TUSparseMatrixPP_h
#define __TUSparseMatrixPP_h

#include "TU/Vector++.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <mkl_pardiso.h>

namespace TU
{
/************************************************************************
*  class SparseMatrix<T, SYM>						*
************************************************************************/
//! Intel Math-Kernel Library(MKL)のフォーマットによる疎行列
template <class T, bool SYM=false>
class SparseMatrix
{
  public:
    typedef T		value_type;		//!< 成分の型

  private:
    template <class S>
    struct Assign : public std::binary_function<T, T, S>
    {
	T	operator ()(const T& x, const S& y)	const	{return y;}
    };
    
  public:
  // 構造生成
    void		beginInit()					;
    void		setRow()					;
    void		setCol(u_int col)				;
    void		setCol(u_int col, value_type val)		;
    void		copyRow()					;
    void		endInit()					;
    
  // 基本情報
    u_int		dim()					const	;
    u_int		nrow()					const	;
    u_int		ncol()					const	;
    u_int		nelements()				const	;
    u_int		nelements(u_int i)			const	;
    T			operator ()(u_int i, u_int j)		const	;
    T&			operator ()(u_int i, u_int j)			;

  // 基本演算
    SparseMatrix&	operator  =(value_type c)			;
    SparseMatrix&	operator *=(value_type c)			;
    SparseMatrix&	operator /=(value_type c)			;
    SparseMatrix&	operator +=(const SparseMatrix& A)		;
    SparseMatrix&	operator -=(const SparseMatrix& A)		;
    SparseMatrix	operator  +(const SparseMatrix& A)	const	;
    SparseMatrix	operator  -(const SparseMatrix& A)	const	;
    template <class S, class B>
    Vector<T>		operator  *(const Vector<S, B>& v)	const	;
    template <class S, class B, class T2, bool SYM2>
    friend Vector<S>	operator  *(const Vector<S, B>& v,
				    const SparseMatrix<T2, SYM2>& A)	;
    SparseMatrix<T, true>
			compose()				const	;
    SparseMatrix<T, true>
			compose(const SparseMatrix<T, true>& W)	const	;

  // ブロック演算
    Vector<T>		operator ()(u_int i, u_int j, u_int d)	const	;
    Matrix<T>		operator ()(u_int i, u_int j,
				    u_int r, u_int c)		const	;
    template <class S, class B>
    SparseMatrix&	assign(u_int i, u_int j, const Vector<S, B>& v)	;
    template <class S, class B, class R>
    SparseMatrix&	assign(u_int i, u_int j,
			       const Matrix<S, B, R>& M)		;
    template <class OP, class S, class B>
    SparseMatrix&	apply(u_int i, u_int j,
			      OP op, const Vector<S, B>& v)		;
    template <class OP, class S, class B, class R>
    SparseMatrix&	apply(u_int i, u_int j,
			      OP op, const Matrix<S, B, R>& M)		;

  // 連立一次方程式
    Vector<T>		solve(const Vector<T>& b)		const	;

  // 入出力
    template <class T2, bool SYM2> friend std::ostream&
			operator <<(std::ostream& out,
				    const SparseMatrix<T2, SYM2>& A)	;
    template <class T2, bool SYM2> friend std::istream&
			operator >>(std::istream& in,
				    SparseMatrix<T2, SYM2>& A)		;

    friend class SparseMatrix<T, !SYM>;
    
  private:
    template <class OP>
    SparseMatrix	binary_op(const SparseMatrix& B, OP op)	const	;
    template <bool SYM2>
    bool		inner_product(const SparseMatrix<T, SYM2>& B,
				      u_int i, u_int j, T& val)	const	;
    int			index(u_int i, u_int j)			const	;
    static int		pardiso_precision()				;

  private:
    u_int		_ncol;		//!< 列の数
    std::vector<int>	_rowIndex;	//!< 各行の先頭成分の通し番号(1ベース)
    std::vector<int>	_columns;	//!< 各成分の列番号(1ベース)
    std::vector<T>	_values;	//!< 各成分の値
};

/*
 * ----------------------- 構造生成 -----------------------------
 */
//! 初期化を開始する．
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::beginInit()
{
    _ncol = 0;
    _rowIndex.clear();
    _columns.clear();
    _values.clear();
}

//! 行の先頭位置をセットする．
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::setRow()
{
  // 1つ前の行の列番号をソート
    if (!_rowIndex.empty())
	std::sort(_columns.begin() + _rowIndex.back() - 1, _columns.end());

  // この行の先頭成分の通し番号をセット(1ベース)
    _rowIndex.push_back(nelements() + 1);
}

//! 成分の列番号をセットする．
/*!
  \param col	列番号
*/
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::setCol(u_int col)
{
    u_int	row = _rowIndex.size();	// 現在までにセットされた行数
    if (row == 0)
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::setCol(): _rowIndex is not set!");
    if (SYM)
    {
	if (col < --row)		// 与えられた列番号を現在の行番号と比較
	    throw std::invalid_argument("TU::SparseMatrix<T, SYM>::setCol(): column index must not be less than row index!");
    }

    _columns.push_back(col + 1);	// この成分の列番号をセット(1ベース)
    if (col + 1 > _ncol)
	_ncol = col + 1;		// 列数を更新
}

//! 成分の列番号とその値をセットする．
/*!
  \param col	列番号
  \param val	値
*/
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::setCol(u_int col, value_type val)
{
    setCol(col);
    _values.push_back(val);
}
    
//! 直前の行と同じ位置に非零成分を持つような行をセットする．
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::copyRow()
{
    u_int	row = _rowIndex.size();	// 現在までにセットされた行数
    if (row == 0)
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::copyRow(): no previous rows!");

  // 行の先頭成分の通し番号をセットする．rowが現在の行番号となる．
    setRow();

  // 直前の行の(対称行列の場合は2番目以降の)成分の列番号を現在の行にコピーする．
    for (u_int n  = _rowIndex[row - 1] - (SYM ? 0 : 1),
	       ne = _rowIndex[row] - 1; n < ne; ++n)
	_columns.push_back(_columns[n]);
}

//! 初期化を完了する．
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::endInit()
{
    setRow();				// ダミーの行をセット

    if (SYM && (nrow() != ncol()))
	throw std::runtime_error("SparseMatrix<T, true>::endInit(): the numbers of rows and columns must be equal!");
    
    if (_values.empty())
    {
	_values.resize(nelements());	// 成分を格納する領域を確保
	operator =(T(0));
    }
    else if (_values.size() != _columns.size())
	throw std::runtime_error("SparseMatrix<T, SYM>::endInit(): the number of values is inconsistent with the number of column indices!");
}

/*
 * ----------------------- 基本情報 ---------------------------------
 */
//! 行列の次元すなわち行数を返す．
/*!
  \return	行列の次元(=行数)
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::dim() const
{
    return nrow();
}
    
//! 行列の行数を返す．
/*!
  \return	行列の行数
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::nrow() const
{
    return _rowIndex.size() - 1;
}
    
//! 行列の列数を返す．
/*!
  \return	行列の列数
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::ncol() const
{
    return _ncol;
}
    
//! 行列の非零成分数を返す．
/*!
  \return	行列の非零成分数
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::nelements() const
{
    return _columns.size();
}
    
//! 指定された行の成分数を返す．
/*!
  \param i	行番号
  \return	第i行の成分数
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::nelements(u_int i) const
{
    return _rowIndex[i+1] - _rowIndex[i];
}
    
//! 指定された行と列に対応する成分を返す．
/*!
  \param i			行番号
  \param j			列番号
  \return			(i, j)成分
*/
template <class T, bool SYM> inline T
SparseMatrix<T, SYM>::operator ()(u_int i, u_int j) const
{
    const int	n = index(i, j);
    return (n >= 0 ? _values[n] : T(0));
}

//! 指定された行と列に対応する成分を返す．
/*!
  \param i			行番号
  \param j			列番号
  \return			(i, j)成分
  \throw std::out_of_range	この行列が(i, j)成分を持たない場合に送出
*/
template <class T, bool SYM> inline T&
SparseMatrix<T, SYM>::operator ()(u_int i, u_int j)
{
    const int	n = index(i, j);
    if (n < 0)
	throw std::out_of_range("TU::SparseMatrix<T, SYM>::operator (): non-existent element!");
    return _values[n];
}
    
/*
 * ----------------------- 基本演算 ---------------------------------
 */
//! すべての0でない成分に定数を代入する．
/*!
  \param c	代入する定数
  \return	この疎行列
*/
template <class T, bool SYM> inline SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator =(value_type c)
{
    std::fill(_values.begin(), _values.end(), c);
    return *this;
}
    
//! この疎行列に定数を掛ける．
/*!
  \param c	掛ける定数
  \return	この疎行列
*/
template <class T, bool SYM> SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator *=(value_type c)
{
    std::transform(_values.begin(), _values.end(), _values.begin(),
		   std::bind2nd(std::multiplies<T>(), c));
    return *this;
}
    
//! この疎行列を定数で割る．
/*!
  \param c	掛ける定数
  \return	この疎行列
*/
template <class T, bool SYM> SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator /=(value_type c)
{
    std::transform(_values.begin(), _values.end(), _values.begin(),
		   std::bind2nd(std::divides<T>(), c));
    return *this;
}
    
//! この疎行列に他の疎行列を足す．
/*!
  2つの疎行列は同一の構造を持たねばならない．
  \param A			足す疎行列
  \return			この疎行列
  \throw std::invalid_argument	2つの疎行列の構造が一致しない場合に送出
*/
template <class T, bool SYM> SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator +=(const SparseMatrix& A)
{
    if (nrow()	    != A.nrow()						||
	nelements() != A.nelements()					||
	!equal(_rowIndex.begin(), _rowIndex.end(), A._rowIndex.begin())	||
	!equal(_columns.begin(),  _columns.end(),  A._columns.begin()))
	throw std::invalid_argument("TU::SparseMatrix<T, SYM>::operator +=(): structure mismatch!");
    std::transform(_values.begin(), _values.end(),
		   A._values.begin(), _values.begin(), std::plus<T>());
    return *this;
}
    
//! この疎行列から他の疎行列を引く．
/*!
  2つの疎行列は同一の構造を持たねばならない．
  \param A			引く疎行列
  \return			この疎行列
  \throw std::invalid_argument	2つの疎行列の構造が一致しない場合に送出
*/
template <class T, bool SYM> SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator -=(const SparseMatrix& A)
{
    if (nrow()	    != A.nrow()						||
	nelements() != A.nelements()					||
	!equal(_rowIndex.begin(), _rowIndex.end(), A._rowIndex.begin())	||
	!equal(_columns.begin(),  _columns.end(),  A._columns.begin()))
	throw std::invalid_argument("TU::SparseMatrix<T, SYM>::operator -=(): structure mismatch!");
    std::transform(_values.begin(), _values.end(),
		   A._values.begin(), _values.begin(), std::minus<T>());
    return *this;
}

//! この疎行列と他の疎行列の和を計算する．
/*!
  2つの疎行列は同一のサイズを持たねばならない．
  \param A	足す疎行列
  \return	2つの疎行列の和
*/
template <class T, bool SYM> inline SparseMatrix<T, SYM>
SparseMatrix<T, SYM>::operator +(const SparseMatrix& A) const
{
    return binary_op(A, std::plus<T>());
}
    
//! この疎行列と他の疎行列の差を計算する．
/*!
  2つの疎行列は同一のサイズを持たねばならない．
  \param A	引く疎行列
  \return	2つの疎行列の差
*/
template <class T, bool SYM> inline SparseMatrix<T, SYM>
SparseMatrix<T, SYM>::operator -(const SparseMatrix& A) const
{
    return binary_op(A, std::minus<T>());
}
    
//! この疎行列に右からベクトルを掛ける．
/*!
  \param v	掛けるベクトル
  \return	結果を格納したベクトル
*/
template <class T, bool SYM> template <class S, class B> Vector<T>
SparseMatrix<T, SYM>::operator *(const Vector<S, B>& v) const
{
    v.check_dim(ncol());
    
    Vector<T>	a(nrow());
    for (u_int i = 0; i < nrow(); ++i)
    {
	if (SYM)
	{
	    for (u_int j = 0; j < i; ++j)
	    {
		const int	n = index(i, j);
		if (n >= 0)
		    a[i] += _values[n] * v[j];
	    }
	}
	
	for (u_int n = _rowIndex[i] - 1; n < _rowIndex[i+1] - 1; ++n)
	    a[i] += _values[n] * v[_columns[n] - 1];
    }

    return a;
}

//! この疎行列に右から自身の転置を掛けた行列を返す．
/*!
  \return	結果を格納した疎対称行列
*/
template <class T, bool SYM> SparseMatrix<T, true>
SparseMatrix<T, SYM>::compose() const
{
    SparseMatrix<T, true>	AAt;	// 結果を格納する疎対称行列

    AAt.beginInit();
    for (u_int i = 0; i < nrow(); ++i)
    {
	AAt.setRow();
	
	for (u_int j = i; j < nrow(); ++j)
	{
	    T	val;
	    if (inner_product(*this, i, j, val))
		AAt.setCol(j, val);
	}
    }
    AAt.endInit();
    
    return AAt;
}
    
//! この疎行列に右から与えられた疎対称行列と自身の転置を掛けた行列を返す．
/*!
  \param W	疎対称行列
  \return	結果を格納した疎対称行列
*/
template <class T, bool SYM> SparseMatrix<T, true>
SparseMatrix<T, SYM>::compose(const SparseMatrix<T, true>& W) const
{
    if (ncol() != W.nrow())
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::compose(): mismatched dimension!");

    SparseMatrix<T, false>	AW;
    AW.beginInit();
    for (u_int i = 0; i < nrow(); ++i)
    {
	std::cerr << i << '/' << nrow() << std::endl;
	
	AW.setRow();

	for (u_int j = 0; j < W.nrow(); ++j)
	{
	    T	val;
	    if (inner_product(W, i, j, val))
		AW.setCol(j, val);
	}
    }
    AW.endInit();

    SparseMatrix<T, true>	AWAt;
    AWAt.beginInit();
    for (u_int i = 0; i < AW.nrow(); ++i)
    {
	AWAt.setRow();

	for (u_int j = i; j < nrow(); ++j)
	{
	    T	val;
	    if (AW.inner_product(*this, i, j, val))
		AWAt.setCol(j, val);
	}
    }
    AWAt.endInit();

    return AWAt;
}

/*
 * ----------------------- ブロック演算 -----------------------------
 */
//! 疎行列の行中の密な部分をベクトルとして取り出す．
/*!
  \param i	起点の行番号
  \param j	起点の列番号
  \param d	取り出す成分数
  \return	取り出した成分を並べたベクトル
*/
template <class T, bool SYM> Vector<T>
SparseMatrix<T, SYM>::operator ()(u_int i, u_int j, u_int d) const
{
    Vector<T>	v(d);
    const T*	p = &(*this)(i, j);
    u_int	dj = 0;
    if (SYM)
    {
	for (u_int je = std::min(i, j + v.dim()); j + dj < je; ++dj)
	{
	    v[dj] = *p;
	    p += (nelements(j + dj) - 1);
	}
    }
    for (; dj < v.dim(); ++dj)
	v[dj] = *p++;
    
    return v;
}

//! 疎行列中の密なブロックを行列として取り出す．
/*!
  \param i	起点の行番号
  \param j	起点の列番号
  \param r	取り出す行数
  \param c	取り出す列数
  \return	取り出した成分を並べた行列
*/
template <class T, bool SYM> Matrix<T>
SparseMatrix<T, SYM>::operator ()(u_int i, u_int j,
				  u_int r, u_int c) const
{
    Matrix<T>	M(r, c);
    for (u_int di = 0; di < M.nrow(); ++di)
	M[di] = (*this)(i + di, j, M.ncol());

    return M;
}

//! 疎行列中の行の密な部分をベクトルとみなして与えられたベクトルを代入する．
/*!
  (i, j)成分を起点とする連続部分に与えられたベクトルを代入する．
  \param i	起点の行番号
  \param j	起点の列番号
  \param v	代入するベクトル
  \return	この疎対称行列
*/
template <class T, bool SYM> template <class S, class B>
inline SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::assign(u_int i, u_int j, const Vector<S, B>& v)
{
    return apply(i, j, Assign<S>(), v);
}
    
//! 疎行列中の密なブロックを行列とみなして与えられた行列を代入する．
/*!
  (i, j)成分を起点とする連続部分に与えられた行列を代入する．
  \param i	起点の行番号
  \param j	起点の列番号
  \param M	代入する行列
  \return	この疎対称行列
*/
template <class T, bool SYM> template <class S, class B, class R>
inline SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::assign(u_int i, u_int j, const Matrix<S, B, R>& M)
{
    return apply(i, j, Assign<S>(), M);
}
    
//! 与えられたベクトルの各成分を指定された成分を起点として適用する．
/*!
  \param i	起点の行番号
  \param j	起点の列番号
  \param f	T型，S型の引数をとりT型の値を返す2項演算子
  \param v	その各成分がfの第2引数となるベクトル
  \return	この疎対称行列
*/
template <class T, bool SYM> template <class OP, class S, class B>
SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::apply(u_int i, u_int j, OP op, const Vector<S, B>& v)
{
    T*		p = &(*this)(i, j);
    u_int	dj = 0;
    if (SYM)
    {
	for (u_int je = std::min(i, j + v.dim()); j + dj < je; ++dj)
	{
	    *p = op(*p, v[dj]);
	    p += (nelements(j + dj) - 1);
	}
    }
    for (; dj < v.dim(); ++dj)
    {
	*p = op(*p, v[dj]);
	++p;
    }
    
    return *this;
}
    
//! 与えられた行列の各成分を指定された成分を起点として適用する．
/*!
  \param i	起点の行番号
  \param j	起点の列番号
  \param f	T型，S型の引数をとりT型の値を返す2項演算子
  \param M	その各成分がfの第2引数となる行列
  \return	この疎対称行列
*/
template <class T, bool SYM> template <class OP, class S, class B, class R>
SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::apply(u_int i, u_int j, OP op, const Matrix<S, B, R>& M)
{
    if (SYM && i > j)
    {
	const Matrix<S>	Mt = M.trns();
	
	for (u_int dj = 0; dj < Mt.nrow(); ++dj)
	{
	    const u_int	di = std::max(j + dj, i) - i;
	    apply(j + dj, i + di, op, Mt[dj](di, Mt.ncol() - di));
	}
    }
    else
    {
	for (u_int di = 0; di < M.nrow(); ++di)
	{
	    const u_int	dj = (SYM ? std::max(i + di, j) - j : 0);
	    apply(i + di, j + dj, op, M[di](dj, M.ncol() - dj));
	}
    }
    
    return *this;
}
    
/*
 * ----------------------- 連立一次方程式 -----------------------------
 */
//! この行列を係数とする連立一次方程式を解く．
/*!
  MKL direct sparse solverによって
  \f$\TUvec{A}{}\TUvec{x}{} = \TUvec{b}{}\f$を解く．
  \param b	ベクトル
  \return	解ベクトル
*/
template <class T, bool SYM> Vector<T>
SparseMatrix<T, SYM>::solve(const Vector<T>& b) const
{
    b.check_dim(ncol());
    
    if (nrow() != ncol())
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::solve(): not a square matrix!");

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
    _INTEGER_t		mtype  = (SYM ? -2 : 11);	// 実対称／実非対称行列
    _INTEGER_t		phase  = 13;	// 行列の解析から反復による細密化まで
    _INTEGER_t		n      = nrow();// 連立一次方程式の式数
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
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::solve(): PARDISO failed!");

  // pardiso 内で使用した全メモリを解放する．
    phase = -1;
    PARDISO(&pt[0], &maxfct, &mnum, &mtype, &phase, &n, (void*)&_values[0],
	    (_INTEGER_t*)&_rowIndex[0], (_INTEGER_t*)&_columns[0],
	    &perm[0], &nrhs, iparm, &msglvl, (void*)&b[0], &x[0], &error);
    if (error != 0)
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::solve(): PARDISO failed to release memory!");

    return x;
}

/*
 * ----------------------- private members -----------------------------
 */
//! この疎行列と他の疎行列の間で成分毎の2項演算を行う．
/*!
  \param B			もう一方の疎行列
  \return			2つの疎行列間の成分毎の2項演算で得られる疎行列
  \throw std::invalid_argument	2つの疎行列のサイズが一致しない場合に送出
*/
template <class T, bool SYM> template <class OP> SparseMatrix<T, SYM>
SparseMatrix<T, SYM>::binary_op(const SparseMatrix& B, OP op) const
{
    if ((nrow() != B.nrow()) || (ncol() != B.ncol()))
	throw std::invalid_argument("SparseMatrix<T, SYM>::binary_op(): two matrices must have equal sizes!");

    SparseMatrix	S;

    S.beginInit();
    for (u_int i = 0; i < nrow(); ++i)
    {
	S.setRow();

	for (u_int m = _rowIndex[i], n = B._rowIndex[i]; ; )
	{
	    const u_int	j = (m <   _rowIndex[i+1] ?   _columns[m-1] - 1
						  :   ncol());
	    const u_int	k = (n < B._rowIndex[i+1] ? B._columns[n-1] - 1
						  : B.ncol());
	    
	    if (j == k)
	    {
		if (j == ncol())
		    break;
		
		S.setCol(j, op(_values[m-1], B._values[n-1]));
		++m;
		++n;
	    }
	    else if (j < k)
	    {
		S.setCol(j, op(_values[m-1], T(0)));
		++m;
	    }
	    else
	    {
		S.setCol(k, op(T(0), B._values[n-1]));
		++n;
	    }
	}
    }
    S.endInit();

    return S;
}
    
//! この疎行列と与えられた疎行列からそれぞれ1行ずつ取り出し，それらの内積を求める．
/*!
  \param B	もう1つの疎行列
  \param i	この疎行列の行番号
  \param j	B の行番号
  \param val	この疎行列の第i行と B の第j行の内積の値が返される
  \return	この疎行列の第i行と B の第j行が列番号を少なくとも
		1つ共有すればtrue, そうでなければ false
*/
template <class T, bool SYM> template <bool SYM2> bool
SparseMatrix<T, SYM>::inner_product(const SparseMatrix<T, SYM2>& B,
				    u_int i, u_int j, T& val) const
{
    if (ncol() != B.ncol())
	throw std::invalid_argument("inner_product(): mismatched dimension!");
    
    bool	exist = false;
    val = T(0);

    if (SYM)
    {
	for (u_int col = 0; col < i; ++col)
	{
	    int	m, n;
	    if ((m = index(i, col)) >= 0 && (n = B.index(j, col)) >= 0)
	    {
		exist = true;
		val += _values[m] * B._values[n];
	    }
	}
    }

    for (u_int m = _rowIndex[i] - 1; m < _rowIndex[i+1] - 1; ++m)
    {
	const int	n = B.index(j, _columns[m] - 1);
	if (n >= 0)
	{
	    exist = true;
	    val += _values[m] * B._values[n];
	}
    }

    return exist;
}

//! 指定された行と列に対応する #_values のindexを返す．
/*!
  #_rowIndex と同様に1ベースのindexを返すので，実際に #_values に
  アクセスする際には返された値から1を引く必要がある．
  \param i	行番号
  \param j	列番号
  \return	(i, j)成分が存在すればそのindex(0ベース), 存在しなければ-1
*/
template <class T, bool SYM> inline int
SparseMatrix<T, SYM>::index(u_int i, u_int j) const
{
    if (SYM && i > j)
	std::swap(i, j);

    ++j;				// 列番号を1ベースに直す．

  // 指定された列番号に対応する成分がこの行にあるか2分法によって調べる．
    for (u_int low = _rowIndex[i]-1, high = _rowIndex[i+1]-1; low != high; )
    {
	u_int	mid = (low + high) / 2;

	if (j < _columns[mid])
	    high = mid;
	else if (j > _columns[mid])
	    low = mid + 1;
	else
	    return mid;
    }

    return -1;				// みつからなければ-1を返す．
}

template<> inline int
SparseMatrix<float,  false>::pardiso_precision()	{return 1;}
template<> inline int
SparseMatrix<float,  true> ::pardiso_precision()	{return 1;}
template<> inline int
SparseMatrix<double, false>::pardiso_precision()	{return 0;}
template<> inline int
SparseMatrix<double, true> ::pardiso_precision()	{return 0;}

/************************************************************************
*  global functions							*
************************************************************************/
template <class S, class B, class T2, bool SYM2> Vector<S>
operator *(const Vector<S, B>& v, const SparseMatrix<T2, SYM2>& A)
{
    v.check_dim(A.nrow());
    
    Vector<S>	a(A.ncol());
    for (u_int j = 0; j < A.ncol(); ++j)
    {
	for (u_int i = 0; i < (SYM2 ? j : A.nrow()); ++i)
	{
	    const int	n = A.index(i, j);
	    if (n >= 0)
		a[j] += v[i] * A._values[n];
	}
	if (SYM2)
	{
	    for (u_int n = A._rowIndex[j] - 1; n < A._rowIndex[j+1] - 1; ++n)
		a[j] += v[A._columns[n] - 1] * A._values[n];
	}
    }

    return a;
}

//! 出力ストリームへ疎行列を書き出し(ASCII)，さらに改行コードを出力する．
/*!
  \param out	出力ストリーム
  \param A	書き出す疎行列
  \return	outで指定した出力ストリーム
*/
template <class T2, bool SYM2> std::ostream&
operator <<(std::ostream& out, const SparseMatrix<T2, SYM2>& A)
{
    using namespace	std;

    u_int	n = 0;
    for (u_int i = 0; i < A.nrow(); ++i)
    {
	if (SYM2)
	{
	    for (u_int j = 0; j < i; ++j)
		out << " *";
	}

	for (u_int j = (SYM2 ? i : 0); j < A.ncol(); ++j)
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

#endif	// ! __TUSparseMatrixPP_h
