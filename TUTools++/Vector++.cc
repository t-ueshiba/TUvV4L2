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
 *  $Id: Vector++.cc,v 1.22 2007-11-26 07:55:48 ueshiba Exp $
 */
#include "TU/Vector++.h"
#include <stdexcept>
#include <iomanip>

namespace TU
{
/************************************************************************
*  class Vector<T, B>							*
************************************************************************/
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
template <class T, class B> Matrix<T, Buf<T> >
Vector<T, B>::skew() const
{
    if (dim() != 3)
	throw std::invalid_argument("TU::Vector<T, B>::skew: dimension must be 3");
    Matrix<T>	r(3, 3);
    r[2][1] = (*this)[0];
    r[0][2] = (*this)[1];
    r[1][0] = (*this)[2];
    r[1][2] = -r[2][1];
    r[2][0] = -r[0][2];
    r[0][1] = -r[1][0];
    return r;
}

/************************************************************************
*  class Matrix<T, B>							*
************************************************************************/
//! この正方行列を全て同一の対角成分値を持つ対角行列にする．
/*!
  \param c	対角成分の値
  \return	この行列，すなわち\f$\TUvec{A}{} \leftarrow \diag(c,\ldots,c)\f$
*/
template <class T, class B> Matrix<T, B>&
Matrix<T, B>::diag(T c)
{
    check_dim(ncol());
    *this = 0;
    for (int i = 0; i < nrow(); ++i)
	(*this)[i][i] = c;
    return *this;
}

//! この行列の転置行列を返す．
/*!
  \return	転置行列，すなわち\f$\TUtvec{A}{}\f$
*/
template <class T, class B> Matrix<T>
Matrix<T, B>::trns() const
{
    Matrix<T> val(ncol(), nrow());
    for (int i = 0; i < nrow(); ++i)
	for (int j = 0; j < ncol(); ++j)
	    val[j][i] = (*this)[i][j];
    return val;
}

//! この行列の小行列式を返す．
/*!
  \param p	元の行列から取り除く行を指定するindex
  \param q	元の行列から取り除く列を指定するindex
  \return	小行列式，すなわち\f$\det\TUvec{A}{pq}\f$
*/
template <class T, class B> T
Matrix<T, B>::det(int p, int q) const
{
    Matrix<T>		d(nrow()-1, ncol()-1);
    for (int i = 0; i < p; ++i)
    {
	for (int j = 0; j < q; ++j)
	    d[i][j] = (*this)[i][j];
	for (int j = q; j < d.ncol(); ++j)
	    d[i][j] = (*this)[i][j+1];
    }
    for (int i = p; i < d.nrow(); ++i)
    {
	for (int j = 0; j < q; ++j)
	    d[i][j] = (*this)[i+1][j];
	for (int j = q; j < d.ncol(); ++j)
	    d[i][j] = (*this)[i+1][j+1];
    }
    return d.det();
}

//! この正方行列のtraceを返す．
/*!
  \return			trace, すなわち\f$\trace\TUvec{A}{}\f$
  \throw std::invalid_argument	正方行列でない場合に送出
*/
template <class T, class B> T
Matrix<T, B>::trace() const
{
    if (nrow() != ncol())
        throw
	  std::invalid_argument("TU::Matrix<T>::trace(): not square matrix!!");
    T	val = 0.0;
    for (int i = 0; i < nrow(); ++i)
	val += (*this)[i][i];
    return val;
}

//! この行列の余因子行列を返す．
/*!
  \return	余因子行列，すなわち
		\f$\TUtilde{A}{} = (\det\TUvec{A}{})\TUinv{A}{}\f$
*/
template <class T, class B> Matrix<T, B>
Matrix<T, B>::adj() const
{
    Matrix<T, B>	val(nrow(), ncol());
    for (int i = 0; i < val.nrow(); ++i)
	for (int j = 0; j < val.ncol(); ++j)
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
template <class T, class B> Matrix<T>
Matrix<T, B>::pinv(T cndnum) const
{
    SVDecomposition<T>	svd(*this);
    Matrix<T>		val(svd.ncol(), svd.nrow());
    
    for (int i = 0; i < svd.diagonal().dim(); ++i)
	if (fabs(svd[i]) * cndnum > fabs(svd[0]))
	    val += (svd.Ut()[i] / svd[i]) % svd.Vt()[i];

    return val;
}

//! この対称行列の固有値と固有ベクトルを返す．
/*!
    \param eval	絶対値の大きい順に並べられた固有値
    \return	各行が固有ベクトルから成る回転行列，すなわち
		\f[
		  \TUvec{A}{}\TUvec{U}{} =
		  \TUvec{U}{}\diag(\lambda_0,\ldots,\lambda_{n-1}),
		  {\hskip 1em}\mbox{where}{\hskip 0.5em}
		  \TUtvec{U}{}\TUvec{U}{} = \TUvec{I}{n},~\det\TUvec{U}{} = 1
		\f]
		なる\f$\TUtvec{U}{}\f$
*/
template <class T, class B> Matrix<T>
Matrix<T, B>::eigen(Vector<T>& eval) const
{
    TriDiagonal<T>	tri(*this);

    tri.diagonalize();
    eval = tri.diagonal();

    return tri.Ut();
}

//! この対称行列の一般固有値と一般固有ベクトルを返す．
/*!
    \param B	もとの行列と同一サイズの正値対称行列
    \param eval	絶対値の大きい順に並べられた一般固有値
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
template <class T, class B> Matrix<T>
Matrix<T, B>::geigen(const Matrix<T>& BB, Vector<T>& eval) const
{
    Matrix<T>	Ltinv = BB.cholesky().inv(), Linv = Ltinv.trns();
    Matrix<T>	Ut = (Linv * (*this) * Ltinv).eigen(eval);
    
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
template <class T, class B> Matrix<T, B>
Matrix<T, B>::cholesky() const
{
    if (nrow() != ncol())
        throw
	    std::invalid_argument("TU::Matrix<T>::cholesky(): not square matrix!!");

    Matrix<T, B>	Lt(*this);
    for (int i = 0; i < nrow(); ++i)
    {
	T d = Lt[i][i];
	if (d <= 0)
	    throw std::runtime_error("TU::Matrix<T>::cholesky(): not positive definite matrix!!");
	for (int j = 0; j < i; ++j)
	    Lt[i][j] = 0;
	Lt[i][i] = d = sqrt(d);
	for (int j = i + 1; j < ncol(); ++j)
	    Lt[i][j] /= d;
	for (int j = i + 1; j < nrow(); ++j)
	    for (int k = j; k < ncol(); ++k)
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
template <class T, class B> Matrix<T, B>&
Matrix<T, B>::normalize()
{
    T	sum = 0.0;
    for (int i = 0; i < nrow(); ++i)
	sum += (*this)[i] * (*this)[i];
    return *this /= sqrt(sum);
}

//! この行列の左から（転置された）回転行列を掛ける．
/*!
    \return	この行列，すなわち
		\f$\TUvec{A}{}\leftarrow\TUtvec{R}{}\TUvec{A}{}\f$
*/
template <class T, class B> Matrix<T, B>&
Matrix<T, B>::rotate_from_left(const Rotation& r)
{
    for (int j = 0; j < ncol(); ++j)
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
template <class T, class B> Matrix<T, B>&
Matrix<T, B>::rotate_from_right(const Rotation& r)
{
    for (int i = 0; i < nrow(); ++i)
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
template <class T, class B> T
Matrix<T, B>::square() const
{
    T	val = 0.0;
    for (int i = 0; i < nrow(); ++i)
	val += (*this)[i] * (*this)[i];
    return val;
}

//! この行列の下半三角部分を上半三角部分にコピーして対称化する．
/*!
    \return	この行列
*/
template <class T, class B> Matrix<T, B>&
Matrix<T, B>::symmetrize()
{
    for (int i = 0; i < nrow(); ++i)
	for (int j = 0; j < i; ++j)
	    (*this)[j][i] = (*this)[i][j];
    return *this;
}

//! この行列の下半三角部分の符号を反転し，上半三角部分にコピーして反対称化する．
/*!
    \return	この行列
*/
template <class T, class B> Matrix<T, B>&
Matrix<T, B>::antisymmetrize()
{
    for (int i = 0; i < nrow(); ++i)
    {
	(*this)[i][i] = 0.0;
	for (int j = 0; j < i; ++j)
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
template <class T, class B> void
Matrix<T, B>::rot2angle(T& theta_x, T& theta_y, T& theta_z) const
{
    using namespace	std;
    
    if (nrow() != 3 || ncol() != 3)
	throw invalid_argument("TU::Matrix<T>::rot2angle: input matrix must be 3x3!!");

    if ((*this)[0][0] == 0.0 && (*this)[0][1] == 0.0)
    {
	theta_x = atan2(-(*this)[2][1], (*this)[1][1]);
	theta_y = ((*this)[0][2] < 0.0 ? M_PI / 2.0 : -M_PI / 2.0);
	theta_z = 0.0;
    }
    else
    {
	theta_x = atan2((*this)[1][2], (*this)[2][2]);
	theta_y = -asin((*this)[0][2]);
	theta_z = atan2((*this)[0][1], (*this)[0][0]);
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
  なる\f$\theta\f$と\f$\TUvec{n}{}\f$がそれぞれ回転角と回転軸となる．
 \param c	回転角のcos値，すなわち\f$\cos\theta\f$を返す．
 \param s	回転角のsin値，すなわち\f$\sin\theta\f$を返す．
 \return	回転軸を表す3次元単位ベクトル，すなわち\f$\TUvec{n}{}\f$
 \throw std::invalid_argument	3x3行列でない場合に送出
*/
template <class T, class B> Vector<T, FixedSizedBuf<T, 3> >
Matrix<T, B>::rot2axis(T& c, T& s) const
{
    if (nrow() != 3 || ncol() != 3)
	throw std::invalid_argument("TU::Matrix<T>::rot2axis: input matrix must be 3x3!!");

  // Compute cosine and sine of rotation angle.
    const T	trace = (*this)[0][0] + (*this)[1][1] + (*this)[2][2];
    c = (trace - 1.0) / 2.0;
    s = sqrt((trace + 1.0)*(3.0 - trace)) / 2.0;

  // Compute rotation axis.
    Vector<T, FixedSizedBuf<T, 3> >	n;
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
  なる\f$\theta\f$と\f$\TUvec{n}{}\f$がそれぞれ回転角と回転軸となる．
 \return			回転角と回転軸を表す3次元ベクトル，すなわち
				\f$\theta\TUvec{n}{}\f$
 \throw invalid_argument	3x3行列でない場合に送出
*/
template <class T, class B> Vector<T, FixedSizedBuf<T, 3u> >
Matrix<T, B>::rot2axis() const
{
    if (nrow() != 3 || ncol() != 3)
	throw std::invalid_argument("TU::Matrix<T>::rot2axis: input matrix must be 3x3!!");

    Vector<T, FixedSizedBuf<T, 3u> >	axis;
    axis[0] = ((*this)[1][2] - (*this)[2][1]) * 0.5;
    axis[1] = ((*this)[2][0] - (*this)[0][2]) * 0.5;
    axis[2] = ((*this)[0][1] - (*this)[1][0]) * 0.5;
    const T	s = sqrt(axis.square());
    if (s + 1.0 == 1.0)		// s << 1 ?
	return axis;
    const T	trace = (*this)[0][0] + (*this)[1][1] + (*this)[2][2];
    if (trace > 1.0)		// cos > 0 ?
	return  axis *= ( asin(s) / s);
    else
	return  axis *= (-asin(s) / s);
}

/************************************************************************
*  class Householder<T>							*
************************************************************************/
template <class T> void
Householder<T>::apply_from_left(Matrix<T>& a, int m)
{
    if (a.nrow() < dim())
	throw std::invalid_argument("TU::Householder<T>::apply_from_left: # of rows of given matrix is smaller than my dimension !!");
    
    T	scale = 0.0;
    for (int i = m+_d; i < dim(); ++i)
	scale += fabs(a[i][m]);
	
    if (scale != 0.0)
    {
	T	h = 0.0;
	for (int i = m+_d; i < dim(); ++i)
	{
	    a[i][m] /= scale;
	    h += a[i][m] * a[i][m];
	}

	const T	s = (a[m+_d][m] > 0.0 ? sqrt(h) : -sqrt(h));
	h	     += s * a[m+_d][m];			// H = u^2 / 2
	a[m+_d][m]   += s;				// m-th col <== u
	    
	for (int j = m+1; j < a.ncol(); ++j)
	{
	    T	p = 0.0;
	    for (int i = m+_d; i < dim(); ++i)
		p += a[i][m] * a[i][j];
	    p /= h;					// p[j] (p' = u'A / H)
	    for (int i = m+_d; i < dim(); ++i)
		a[i][j] -= a[i][m] * p;			// A = A - u*p'
	    a[m+_d][j] = -a[m+_d][j];
	}
	    
	for (int i = m+_d; i < dim(); ++i)
	    (*this)[m][i] = scale * a[i][m];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::apply_from_right(Matrix<T>& a, int m)
{
    if (a.ncol() < dim())
	throw std::invalid_argument("Householder<T>::apply_from_right: # of column of given matrix is smaller than my dimension !!");
    
    T	scale = 0.0;
    for (int j = m+_d; j < dim(); ++j)
	scale += fabs(a[m][j]);
	
    if (scale != 0.0)
    {
	T	h = 0.0;
	for (int j = m+_d; j < dim(); ++j)
	{
	    a[m][j] /= scale;
	    h += a[m][j] * a[m][j];
	}

	const T	s = (a[m][m+_d] > 0.0 ? sqrt(h) : -sqrt(h));
	h	     += s * a[m][m+_d];			// H = u^2 / 2
	a[m][m+_d]   += s;				// m-th row <== u

	for (int i = m+1; i < a.nrow(); ++i)
	{
	    T	p = 0.0;
	    for (int j = m+_d; j < dim(); ++j)
		p += a[i][j] * a[m][j];
	    p /= h;					// p[i] (p = Au / H)
	    for (int j = m+_d; j < dim(); ++j)
		a[i][j] -= p * a[m][j];			// A = A - p*u'
	    a[i][m+_d] = -a[i][m+_d];
	}
	    
	for (int j = m+_d; j < dim(); ++j)
	    (*this)[m][j] = scale * a[m][j];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::apply_from_both(Matrix<T>& a, int m)
{
    Vector<T>		u = a[m](m+_d, a.ncol()-m-_d);
    T		scale = 0.0;
    for (int j = 0; j < u.dim(); ++j)
	scale += fabs(u[j]);
	
    if (scale != 0.0)
    {
	u /= scale;

	T		h = u * u;
	const T	s = (u[0] > 0.0 ? sqrt(h) : -sqrt(h));
	h	     += s * u[0];			// H = u^2 / 2
	u[0]	     += s;				// m-th row <== u

	Matrix<T>	A = a(m+_d, m+_d, a.nrow()-m-_d, a.ncol()-m-_d);
	Vector<T>	p = _sigma(m+_d, dim()-m-_d);
	for (int i = 0; i < A.nrow(); ++i)
	    p[i] = (A[i] * u) / h;			// p = Au / H

	const T	k = (u * p) / (h + h);		// K = u*p / 2H
	for (int i = 0; i < A.nrow(); ++i)
	{				// m-th col of 'a' is used as 'q'
	    a[m+_d+i][m] = p[i] - k * u[i];		// q = p - Ku
	    for (int j = 0; j <= i; ++j)		// A = A - uq' - qu'
		A[j][i] = (A[i][j] -= (u[i]*a[m+_d+j][m] + a[m+_d+i][m]*u[j]));
	}
	for (int j = 1; j < A.nrow(); ++j)
	    A[j][0] = A[0][j] = -A[0][j];

	for (int j = m+_d; j < a.ncol(); ++j)
	    (*this)[m][j] = scale * a[m][j];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::make_transformation()
{
    for (int m = dim(); --m >= 0; )
    {
	for (int i = m+1; i < dim(); ++i)
	    (*this)[i][m] = 0.0;

	if (_sigma[m] != 0.0)
	{
	    for (int i = m+1; i < dim(); ++i)
	    {
		T	g = 0.0;
		for (int j = m+1; j < dim(); ++j)
		    g += (*this)[i][j] * (*this)[m-_d][j];
		g /= (_sigma[m] * (*this)[m-_d][m]);	// g[i] (g = Uu / H)
		for (int j = m; j < dim(); ++j)
		    (*this)[i][j] -= g * (*this)[m-_d][j];	// U = U - gu'
	    }
	    for (int j = m; j < dim(); ++j)
		(*this)[m][j] = (*this)[m-_d][j] / _sigma[m];
	    (*this)[m][m] -= 1.0;
	}
	else
	{
	    for (int j = m+1; j < dim(); ++j)
		(*this)[m][j] = 0.0;
	    (*this)[m][m] = 1.0;
	}
    }
}

template <class T> bool
Householder<T>::sigma_is_zero(int m, T comp) const
{
    return (T(fabs(_sigma[m])) + comp == comp);
}

/************************************************************************
*  class TriDiagonal<T>							*
************************************************************************/
//! 3重対角行列を対角化する（固有値，固有ベクトルの計算）．
/*!
  対角成分は固有値となり，\f$\TUtvec{U}{}\f$の各行は固有ベクトルを与える．
  \throw std::runtime_error	指定した繰り返し回数を越えた場合に送出
*/ 
template <class T> void
TriDiagonal<T>::diagonalize()
{
    using namespace	std;
    
    for (int n = dim(); --n >= 0; )
    {
	int	niter = 0;
	
#ifdef TUVectorPP_DEBUG
	cerr << "******** n = " << n << " ********" << endl;
#endif
	while (!off_diagonal_is_zero(n))
	{					// n > 0 here
	    if (niter++ > NITER_MAX)
		throw runtime_error("TU::TriDiagonal::diagonalize(): Number of iteration exceeded maximum value!!");

	  /* Find first m (< n) whose off-diagonal element is 0 */
	    int	m = n;
	    while (!off_diagonal_is_zero(--m));	// 0 <= m < n < dim() here

	  /* Set x and y which determine initial(i = m+1) plane rotation */
	    T	x, y;
	    initialize_rotation(m, n, x, y);
	  /* Apply rotation P(i-1, i) for each i (i = m+1, n+2, ... , n) */
	    for (int i = m; ++i <= n; )
	    {
		Rotation	rot(i-1, i, x, y);
		
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
#ifdef TUVectorPP_DEBUG
	    cerr << "  niter = " << niter << ": " << off_diagonal();
#endif	    
	}
    }

    for (int m = 0; m < dim(); ++m)	// sort eigen values and eigen vectors
	for (int n = m+1; n < dim(); ++n)
	    if (fabs(_diagonal[n]) > fabs(_diagonal[m]))
	    {
		swap(_diagonal[m], _diagonal[n]);
		for (int j = 0; j < dim(); ++j)
		{
		    const T	tmp = _Ut[m][j];
		    _Ut[m][j] = _Ut[n][j];
		    _Ut[n][j] = -tmp;
		}
	    }
}

template <class T> bool
TriDiagonal<T>::off_diagonal_is_zero(int n) const
{
    return (n == 0 || _Ut.sigma_is_zero(n, fabs(_diagonal[n-1]) +
					   fabs(_diagonal[n])));
}

template <class T> void
TriDiagonal<T>::initialize_rotation(int m, int n, T& x, T& y) const
{
    const T	g = (_diagonal[n] - _diagonal[n-1]) /
			    (2.0*_off_diagonal[n]),
			absg = fabs(g),
			gg1 = (absg > 1.0 ?
			       absg * sqrt(1.0 + (1.0/absg)*(1.0/absg)) :
			       sqrt(1.0 + absg*absg)),
			t = (g > 0.0 ? g + gg1 : g - gg1);
    x = _diagonal[m] - _diagonal[n] - _off_diagonal[n]/t;
  //x = _diagonal[m];					// without shifting
    y = _off_diagonal[m+1];
}

/************************************************************************
*  class BiDiagonal<T>							*
************************************************************************/
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
    
    for (int n = _Et.dim(); --n >= 0; )
    {
	int	niter = 0;
	
#ifdef TUVectorPP_DEBUG
	cerr << "******** n = " << n << " ********" << endl;
#endif
	while (!off_diagonal_is_zero(n))	// n > 0 here
	{
	    if (niter++ > NITER_MAX)
		throw runtime_error("TU::BiDiagonal::diagonalize(): Number of iteration exceeded maximum value");
	    
	  /* Find first m (< n) whose off-diagonal element is 0 */
	    int m = n;
	    do
	    {
		if (diagonal_is_zero(m-1))
		{ // If _diagonal[m-1] is zero, make _off_diagonal[m] zero.
		    T	x = _diagonal[m], y = _off_diagonal[m];
		    _off_diagonal[m] = 0.0;
		    for (int i = m; i <= n; ++i)
		    {
			Rotation	rotD(m-1, i, x, -y);

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
#ifdef TUBiDiagonal_DEBUG
	    cerr << "--- m = " << m << ", n = " << n << "---"
		 << endl;
	    cerr << "  diagonal:     " << diagonal();
	    cerr << "  off-diagonal: " << off_diagonal();
#endif
	  /* Apply rotation P(i-1, i) for each i (i = m+1, n+2, ... , n) */
	    for (int i = m; ++i <= n; )
	    {
	      /* Apply rotation from left */
		Rotation	rotE(i-1, i, x, y);
		
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
		Rotation	rotD(i-1, i, x, y);

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
#ifdef TUVectorPP_DEBUG
	    cerr << "  niter = " << niter << ": " << off_diagonal();
#endif
	}
    }

    for (int m = 0; m < _Et.dim(); ++m)  // sort singular values and vectors
	for (int n = m+1; n < _Et.dim(); ++n)
	    if (fabs(_diagonal[n]) > fabs(_diagonal[m]))
	    {
		swap(_diagonal[m], _diagonal[n]);
		for (int j = 0; j < _Et.dim(); ++j)
		{
		    const T	tmp = _Et[m][j];
		    _Et[m][j] = _Et[n][j];
		    _Et[n][j] = -tmp;
		}
		for (int j = 0; j < _Dt.dim(); ++j)
		{
		    const T	tmp = _Dt[m][j];
		    _Dt[m][j] = _Dt[n][j];
		    _Dt[n][j] = -tmp;
		}
	    }

    int l = _Et.dim() - 1;		// last index
    for (int m = 0; m < l; ++m)		// ensure positivity of all singular
	if (_diagonal[m] < 0.0)		// values except for the last one.
	{
	    _diagonal[m] = -_diagonal[m];
	    _diagonal[l] = -_diagonal[l];
	    for (int j = 0; j < _Et.dim(); ++j)
	    {
		_Et[m][j] = -_Et[m][j];
		_Et[l][j] = -_Et[l][j];
	    }
	}
}

template <class T> bool
BiDiagonal<T>::diagonal_is_zero(int n) const
{
    return _Dt.sigma_is_zero(n, _anorm);
}

template <class T> bool
BiDiagonal<T>::off_diagonal_is_zero(int n) const
{
    return _Et.sigma_is_zero(n, _anorm);
}

template <class T> void
BiDiagonal<T>::initialize_rotation(int m, int n, T& x, T& y) const
{
    const T	g = ((_diagonal[n]     + _diagonal[n-1])*
		     (_diagonal[n]     - _diagonal[n-1])+
		     (_off_diagonal[n] + _off_diagonal[n-1])*
		     (_off_diagonal[n] - _off_diagonal[n-1]))
		  / (2.0*_diagonal[n-1]*_off_diagonal[n]),
      // Caution!! You have to ensure that _diagonal[n-1] != 0
      // as well as _off_diagonal[n].
		absg = fabs(g),
		gg1 = (absg > 1.0 ?
		       absg * sqrt(1.0 + (1.0/absg)*(1.0/absg)) :
		       sqrt(1.0 + absg*absg)),
		t = (g > 0.0 ? g + gg1 : g - gg1);
    x = ((_diagonal[m] + _diagonal[n])*(_diagonal[m] - _diagonal[n]) -
	 _off_diagonal[n]*(_off_diagonal[n] + _diagonal[n-1]/t)) / _diagonal[m];
  //x = _diagonal[m];				// without shifting
    y = _off_diagonal[m+1];
}

/************************************************************************
*  class Minimization1<S>						*
************************************************************************/
/*
 *  Minimize 1-dimensional function using golden section search and minima
 *  is returned in x. Minimum value of the func is also returned as a return
 *  value.
 */
template <class S> S
Minimization1<S>::minimize(S& x, S w) const
{
#define W	0.38197
    using namespace	std;

    S	x1 = x, x2 = x + w, f1 = (*this)(x1), f2 = (*this)(x2);
    
    if (f1 < f2)			// guarantee that f1 >= f2
    {
	S	tmp = x1;		// swap x1 & x2
	x1  = x2;
	x2  = tmp;
	tmp = f1;			// swap f1 & f2
	f1  = f2;
	f2  = tmp;
    }
    S	x0;
    do
    {
	x0  = x1;
	x1  = x2;
	x2 += (1.0 / W - 1.0) * (x1 - x0);	// elongate to right
#ifdef MIN1_DEBUG
	S	f0 = f1;
#endif
	f1  = f2;
	f2  = (*this)(x2);
#ifdef MIN1_DEBUG
	std::cerr << "Bracketting: [" << x0 << ", " << x1 << ", " << x2
		  << "], (" << f0 << ", " << f1 << ", " << f2 << ")"
		  << std::endl;
#endif
    } while (f1 > f2);
    
  /* Golden section search */
    S	x3 = x2;
    if (fabs(x1 - x0) > fabs(x2 - x1))
    {
	x2  = x1;
	x1 -= W * (x2 - x0);		// insert new x1 between x0 & x2
	f2  = f1;
	f1  = (*this)(x1);
    }
    else
    {
	x2 -= (1.0 - W) * (x3 - x1);	// insert new x2 between x1 & x3
	f2  = (*this)(x2);
    }
#ifdef MIN1_DEBUG
    std::cerr << "Initial:     [" << x0 << ", " << x1 << ", " << x2
	      << ", " << x3 << "], (" << f1 << ", " << f2 << ")" << std::endl;
#endif
    int	i;
    for (i = 0;
	 i < _niter_max && fabs(x3 - x0) > _tol * (fabs(x1) + fabs(x2)); ++i)
    {
	if (f1 < f2)
	{
	    x3  = x2;			// shift x2 & x3 to left
	    x2  = x1;
	    x1 -= W * (x2 - x0);	// insert new x1 between x0 & x2
	    f2  = f1;
	    f1  = (*this)(x1);
	}
	else
	{
	    x0  = x1;			// shift x0 & x1 to right
	    x1  = x2;
	    x2 += W * (x3 - x1);	// insert new x2 between x1 & x3
	    f1  = f2;
	    f2  = (*this)(x2);
	}
#ifdef MIN1_DEBUG
	std::cerr << "Golden:      [" << x0 << ", " << x1 << ", " << x2
		  << ", " << x3 << "], (" << f1 << ", " << f2 << ")"
		  << std::endl;
#endif
    }
    if (i == _niter_max)
	throw std::runtime_error("TU::Minimization1<S>::minimize(): Too many iterations!!");

    if (f1 < f2)
    {
	x = x1;
	return f1;
    }
    else
    {
	x = x2;
	return f2;
    }
}

/************************************************************************
*  class Minimization<S, T>						*
************************************************************************/
/*
 *  Minimize multi-dimensional function using conjugate gradient method and
 *  minima is returned in x. Minimum value of the func is also returned as
 *  a return value.
 */
template <class S, class T> S
Minimization<S, T>::minimize(T& x)
{
    S		val = (*this)(x);
    Vector<S>	g   = ngrad(x), h = g;
    
    for (int i = 0; i < _niter_max; ++i)
    {
	if (_print)
	    print(i, val, x);

	const S		g_sqr = g * g;
	if (g_sqr == 0.0)
	    return val;

	const S		val_next = line_minimize(x, h);
	if (near_enough(val, val_next))
	    return val_next;
	val = val_next;

	const Vector<S>	g_next = ngrad(x);
	h = g_next + (((g_next - g) * g_next) / g_sqr) * h;
	g = g_next;
	update(x);
    }

    std::cerr << "TU::Minimization<S, T>::minimize(): Too many iterations!!"
	      << std::endl;
    return val;
}

/*
 *  Minimize multi-dimensional function using steepest descent method and
 *  minima is returned in x. Minimum value of the func is also returned as
 *  a return value.
 */
template <class S, class T> S
Minimization<S, T>::steepest_descent(T& x)
{
    S		val = (*this)(x);
    Vector<S>	g   = ngrad(x);
    
    for (int i = 0; i < _niter_max; ++i)
    {
	if (_print)
	    print(i, val, x);
	
	const S		g_sqr = g * g;
	if (g_sqr == 0.0)
	    return val;

	const S		val_next = line_minimize(x, g);
	if (near_enough(val, val_next))
	    return val_next;
	val = val_next;

	g = ngrad(x);
	update(x);
    }

    std::cerr << "TU::Minimization<S, T>::steepest_descent(): Too many iterations!!"
	      << std::endl;
    return val;
}

/*
 *  Minimize function along direction h and minima is returned in x.
 *  Minimum value of the function is also returned as a return value.
 */
template <class S, class T> S
Minimization<S, T>::line_minimize(T &x, const Vector<S>& h) const
{
    LineFunction	lfunc(*this, x, h, _tol, _niter_max);
    S			d = 0.0, val = lfunc.minimize(d, 1.0);
    x = proceed(x, d * h);
    return val;
}

/*
 *  Update the status of the function to be minimized.
 */
template <class S, class T> void
Minimization<S, T>::update(const T&)
{
}

/*
 *  Print intermediate values
 */
template <class S, class T> void
Minimization<S, T>::print(int i, S val, const T& x) const
{
    std::cerr << std::setw(3) << i << ": (" << val << ')' << x;
}
 
}
