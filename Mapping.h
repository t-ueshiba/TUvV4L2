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
 *  $Id: Mapping.h,v 1.7 2009-09-04 04:01:06 ueshiba Exp $
 */
#ifndef __TUMapping_h
#define __TUMapping_h

#include "TU/utility.h"
#include "TU/Normalize.h"
#include "TU/Minimize.h"

namespace TU
{
/************************************************************************
*  class ProjectiveMapping						*
************************************************************************/
//! 射影変換を行うクラス
/*!
  \f$\TUvec{T}{} \in \TUspace{R}{(n+1)\times(m+1)}\f$を用いてm次元空間の点
  \f$\TUud{x}{} \in \TUspace{R}{m+1}\f$をn次元空間の点
  \f$\TUud{y}{} \simeq \TUvec{T}{}\TUud{x}{} \in \TUspace{R}{n+1}\f$
  に写す（\f$m \neq n\f$でも構わない）．
*/
class __PORT ProjectiveMapping
{
  public:
    typedef double	value_type;

  public:
    ProjectiveMapping(u_int inDim=2, u_int outDim=2)			;

  //! 変換行列を指定して射影変換オブジェクトを生成する．
  /*!
    \param T			(m+1)x(n+1)行列（m, nは入力／出力空間の次元）
  */
    ProjectiveMapping(const Matrix<double>& T)	:_T(T)			{}

    template <class Iterator>
    ProjectiveMapping(Iterator first, Iterator last, bool refine=false)	;

  //! 変換行列を指定する．
  /*!
    \param T			(m+1)x(n+1)行列（m, nは入力／出力空間の次元）
  */
    void		set(const Matrix<double>& T)	{_T = T;}
    
    template <class Iterator>
    void		fit(Iterator first, Iterator last,
			    bool refine=false)				;

  //! この射影変換の入力空間の次元を返す．
  /*! 
    \return	入力空間の次元(同次座標のベクトルとしての次元は#inDim()+1)
  */
    u_int		inDim()			const	{return _T.ncol()-1;}

  //! この射影変換の出力空間の次元を返す．
  /*! 
    \return	出力空間の次元(同次座標のベクトルとしての次元は#outDim()+1)
  */
    u_int		outDim()		const	{return _T.nrow()-1;}

    u_int		ndataMin()		const	;
    
  //! この射影変換を表現する行列を返す．
  /*! 
    \return	(#outDim()+1)x(#inDim()+1)行列
  */
    const Matrix<double>&	T()		const	{return _T;}

    template <class S, class B>
    Vector<double>	operator ()(const Vector<S, B>& x)	const	;
    template <class S, class B>
    Vector<double>	mapP(const Vector<S, B>& x)		const	;
    template <class S, class B>
    Matrix<double>	jacobian(const Vector<S, B>& x)		const	;

    template <class In, class Out>
    double		sqdist(const std::pair<In, Out>& pair)	const	;
    template <class In, class Out>
    double		dist(const std::pair<In, Out>& pair)	const	;
    double		square()				const	;
			operator const Vector<double>()		const	;
    u_int		dof()					const	;
    void		update(const Vector<double>& dt)		;
    
  protected:
    Matrix<double>	_T;			//!< 射影変換を表現する行列

  protected:
  //! 射影変換行列の最尤推定のためのコスト関数
    template <class AT, class Iterator>
    class Cost
    {
      public:
	typedef double	value_type;

	Cost(Iterator first, Iterator last)				;

	Vector<double>	operator ()(const AT& map)		const	;
	Matrix<double>	jacobian(const AT& map)			const	;
	static void	update(AT& map, const Vector<double>& dm)	;
	u_int		npoints()				const	;

      private:
	const Iterator	_first, _last;
	const u_int	_npoints;
    };
};

//! 与えられた点対列の非同次座標から射影変換オブジェクトを生成する．
/*!
  \param first			点対列の先頭を示す反復子
  \param last			点対列の末尾を示す反復子
  \param refine			非線型最適化の有(true)／無(false)を指定
  \throw std::invalid_argument	点対の数が#ndataMin()に満たない場合に送出
*/
template <class Iterator> inline
ProjectiveMapping::ProjectiveMapping(Iterator first, Iterator last,
				     bool refine)
{
    fit(first, last, refine);
}

//! 与えられた点対列の非同次座標から射影変換を計算する．
/*!
  \param first			点対列の先頭を示す反復子
  \param last			点対列の末尾を示す反復子
  \param refine			非線型最適化の有(true)／無(false)を指定
  \throw std::invalid_argument	点対の数が#ndataMin()に満たない場合に送出
*/
template <class Iterator> void
ProjectiveMapping::fit(Iterator first, Iterator last, bool refine)
{
  // 点列の正規化
    const Normalize	xNormalize(make_const_first_iterator(first),
				   make_const_first_iterator(last)),
			yNormalize(make_const_second_iterator(first),
				   make_const_second_iterator(last));

  // 充分な個数の点対があるか？
    const u_int		ndata = std::distance(first, last);
    const u_int		xdim1 = xNormalize.spaceDim() + 1,
			ydim  = yNormalize.spaceDim();
    if (ndata*ydim < xdim1*(ydim + 1) - 1)	// _Tのサイズが未定なので
						// ndataMin()は無効
	throw std::invalid_argument("ProjectiveMapping::fit(): not enough input data!!");

  // データ行列の計算
    Matrix<double>	A(xdim1*(ydim + 1), xdim1*(ydim + 1));
    for (Iterator iter = first; iter != last; ++iter)
    {
	const Vector<double>&	x  = xNormalize.normalizeP(iter->first);
	const Vector<double>&	y  = yNormalize(iter->second);
	const Matrix<double>&	xx = x % x;
	A(0, 0, xdim1, xdim1) += xx;
	for (int j = 0; j < ydim; ++j)
	    A(ydim*xdim1, j*xdim1, xdim1, xdim1) -= y[j] * xx;
	A(ydim*xdim1, ydim*xdim1, xdim1, xdim1) += (y*y) * xx;
    }
    for (int j = 1; j < ydim; ++j)
	A(j*xdim1, j*xdim1, xdim1, xdim1) = A(0, 0, xdim1, xdim1);
    A.symmetrize();

  // データ行列の最小固有値に対応する固有ベクトルから変換行列を計算し，
  // 正規化をキャンセルする．
    Vector<double>	eval;
    Matrix<double>	Ut = A.eigen(eval);
    _T = yNormalize.Tinv()
       * Matrix<double>((double*)Ut[Ut.nrow()-1], ydim + 1, xdim1)
       * xNormalize.T();

  // 変換行列が正方ならば，その行列式が１になるように正規化する．
    if (_T.nrow() == _T.ncol())
    {
	double	det = _T.det();
	if (det > 0)
	    _T /= pow(det, 1.0/_T.nrow());
	else
	    _T /= -pow(-det, 1.0/_T.nrow());
    }

  // 非線型最適化を行う．
    if (refine)
    {
	Cost<ProjectiveMapping, Iterator>	cost(first, last);
	ConstNormConstraint<ProjectiveMapping>	constraint(*this);
	minimizeSquare(cost, constraint, *this);
    }
}

//! 射影変換を求めるために必要な点対の最小個数を返す．
/*!
  現在設定されている入出力空間の次元をもとに計算される．
  \return	必要な点対の最小個数すなわち入力空間の次元m，出力空間の次元n
		に対して m + 1 + m/n
*/
inline u_int
ProjectiveMapping::ndataMin() const
{
    return inDim() + 1 + u_int(ceil(double(inDim())/double(outDim())));
}
    
//! 与えられた点に射影変換を適用してその非同次座標を返す．
/*!
  \param x	点の非同次座標（#inDim()次元）または同次座標（#inDim()+1次元）
  \return	射影変換された点の非同次座標（#outDim()次元）
*/
template <class S, class B> inline Vector<double>
ProjectiveMapping::operator ()(const Vector<S, B>& x) const
{
    return mapP(x).inhomogenize();
}

//! 与えられた点に射影変換を適用してその同次座標を返す．
/*!
  \param x	点の非同次座標（#inDim()次元）または同次座標（#inDim()+1次元）
  \return	射影変換された点の同次座標（#outDim()+1次元）
*/
template <class S, class B> inline Vector<double>
ProjectiveMapping::mapP(const Vector<S, B>& x) const
{
    if (x.dim() == inDim())
	return _T * x.homogenize();
    else
	return _T * x;
}

//! 与えられた点におけるヤコビ行列を返す．
/*!
  ヤコビ行列とは射影変換行列成分に関する1階微分のことである．
  \param x	点の非同次座標（#inDim()次元）または同次座標（#inDim()+1次元）
  \return	#outDim() x (#outDim()+1)x(#inDim()+1)ヤコビ行列
*/
template <class S, class B> Matrix<double>
ProjectiveMapping::jacobian(const Vector<S, B>& x) const
{
    Vector<double>		xP;
    if (x.dim() == inDim())
	xP = x.homogenize();
    else
	xP = x;
    const Vector<double>&	y  = mapP(xP);
    Matrix<double>		J(outDim(), (outDim() + 1)*xP.dim());
    for (int i = 0; i < J.nrow(); ++i)
    {
	J[i](i*xP.dim(), xP.dim()) = xP;
	(J[i](outDim()*xP.dim(), xP.dim()) = xP) *= (-y[i]/y[outDim()]);
    }
    J /= y[outDim()];

    return J;
}
    
//! 入力点に射影変換を適用した点と出力点の距離の2乗を返す．
/*!
  \param pair	入力点の非同次座標（#inDim()次元）と出力点の非同次座標
		（#outDim()次元）の対
  \return	変換された入力点と出力点の距離の2乗
*/
template <class In, class Out> inline double
ProjectiveMapping::sqdist(const std::pair<In, Out>& pair) const
{
    return (*this)(pair.first).sqdist(pair.second);
}
    
//! 入力点に射影変換を適用した点と出力点の距離を返す．
/*!
  \param pair	入力点の非同次座標（#inDim()次元）と出力点の非同次座標
		（#outDim()次元）の対
  \return	変換された入力点と出力点の距離
*/
template <class In, class Out> inline double
ProjectiveMapping::dist(const std::pair<In, Out>& pair) const
{
    return sqrt(sqdist(pair));
}

//! 射影変換行列のノルムの2乗を返す．
/*!
  \return	射影変換行列のノルムの2乗
*/
inline double
ProjectiveMapping::square() const
{
    return _T.square();
}

//! 射影変換行列の各行を順番に並べたベクトルを返す．
/*!
  \return	#T()の成分を並べたベクトル（(#outDim()+1)x(#inDim()+1)次元）
*/
inline
ProjectiveMapping::operator const Vector<double>() const
{
    return Vector<double>(const_cast<Matrix<double>&>(_T));
}

//! この射影変換の自由度を返す．
/*!
  \return	射影変換の自由度（(#outDim()+1)x(#inDim()+1)-1）
*/
inline u_int
ProjectiveMapping::dof() const
{
    return (outDim() + 1)*(inDim() + 1)-1;
}

//! 射影変換行列を与えられた量だけ修正する．
/*!
  \param dt	修正量を表すベクトル（(#outDim()+1)x(#inDim()+1)次元）
*/
inline void
ProjectiveMapping::update(const Vector<double>& dt)
{
    Vector<double>	t(_T);
    double		l = t.length();
    (t -= dt).normalize() *= l;
}
    
template <class AT, class Iterator>
ProjectiveMapping::Cost<AT, Iterator>::Cost(Iterator first, Iterator last)
    :_first(first), _last(last), _npoints(std::distance(_first, _last))
{
}
    
template <class AT, class Iterator> Vector<double>
ProjectiveMapping::Cost<AT, Iterator>::operator ()(const AT& map) const
{
    const u_int		outDim = map.outDim();
    Vector<double>	val(_npoints*outDim);
    int			n = 0;
    for (Iterator iter = _first; iter != _last; ++iter)
    {
	val(n, outDim) = map(iter->first) - iter->second;
	n += outDim;
    }
    
    return val;
}
    
template <class AT, class Iterator> Matrix<double>
ProjectiveMapping::Cost<AT, Iterator>::jacobian(const AT& map) const
{
    const u_int		outDim = map.outDim();
    Matrix<double>	J(_npoints*outDim, map.dof()+1);
    int			n = 0;
    for (Iterator iter = _first; iter != _last; ++iter)
    {
	J(n, 0, outDim, J.ncol()) = map.jacobian(iter->first);
	n += outDim;
    }

    return J;
}

template <class AT, class Iterator> inline void
ProjectiveMapping::Cost<AT, Iterator>::update(AT& map, const Vector<double>& dm)
{
    map.update(dm);
}
    
/************************************************************************
*  class AffineMapping							*
************************************************************************/
//! アフィン変換を行うクラス
/*!
  \f$\TUvec{A}{} \in \TUspace{R}{n\times m}\f$と
  \f$\TUvec{b}{} \in \TUspace{R}{n}\f$を用いてm次元空間の点
  \f$\TUvec{x}{} \in \TUspace{R}{m}\f$をn次元空間の点
  \f$\TUvec{y}{} \simeq \TUvec{A}{}\TUvec{x}{} + \TUvec{b}{}
  \in \TUspace{R}{n}\f$に写す（\f$m \neq n\f$でも構わない）．
*/
class __PORT AffineMapping : public ProjectiveMapping
{
  public:
  //! 入力空間と出力空間の次元を指定してアフィン変換オブジェクトを生成する．
  /*!
    恒等変換として初期化される．
    \param inDim	入力空間の次元
    \param outDim	出力空間の次元
  */
    AffineMapping(u_int inDim=2, u_int outDim=2)
	:ProjectiveMapping(inDim, outDim)				{}

    AffineMapping(const Matrix<double>& T)				;
    template <class Iterator>
    AffineMapping(Iterator first, Iterator last)			;

    void	set(const Matrix<double>& T)				;
    template <class Iterator>
    void	fit(Iterator first, Iterator last)			;
    u_int	ndataMin()					const	;
    
  //! このアフィン変換の変形部分を表現する行列を返す．
  /*! 
    \return	#outDim() x #inDim()行列
  */
    const Matrix<double>
			A()	const	{return _T(0, 0, outDim(), inDim());}
    
    Vector<double>	b()	const	;
};

//! 変換行列を指定してアフィン変換オブジェクトを生成する．
/*!
  変換行列の下端行は強制的に 0,0,...,0,1 に設定される．
  \param T			(m+1)x(n+1)行列（m, nは入力／出力空間の次元）
*/
inline
AffineMapping::AffineMapping(const Matrix<double>& T)
    :ProjectiveMapping(T)
{
    _T[outDim()]	  = 0.0;
    _T[outDim()][inDim()] = 1.0;
}

//! 与えられた点対列の非同次座標からアフィン変換オブジェクトを生成する．
/*!
  \param first			点対列の先頭を示す反復子
  \param last			点対列の末尾を示す反復子
  \throw std::invalid_argument	点対の数が#ndataMin()に満たない場合に送出
*/
template <class Iterator> inline
AffineMapping::AffineMapping(Iterator first, Iterator last)
{
    fit(first, last);
}

//! 変換行列を指定する．
/*!
  変換行列の下端行は強制的に 0,0,...,0,1 に設定される．
  \param T			(m+1)x(n+1)行列（m, nは入力／出力空間の次元）
*/
inline void
AffineMapping::set(const Matrix<double>& T)
{
    ProjectiveMapping::set(T);
    _T[outDim()]	  = 0.0;
    _T[outDim()][inDim()] = 1.0;
}
    
//! 与えられた点対列の非同次座標からアフィン変換を計算する．
/*!
  \param first			点対列の先頭を示す反復子
  \param last			点対列の末尾を示す反復子
  \throw std::invalid_argument	点対の数が#ndataMin()に満たない場合に送出
*/
template <class Iterator> void
AffineMapping::fit(Iterator first, Iterator last)
{
  // 充分な個数の点対があるか？
    const u_int		ndata = std::distance(first, last);
    if (ndata == 0)		// firstが有効か？
	throw std::invalid_argument("AffineMapping::fit(): 0-length input data!!");
    const u_int		xdim = first->first.dim();
    if (ndata < xdim + 1)	// _Tのサイズが未定なのでndataMin()は無効
	throw std::invalid_argument("AffineMapping::fit(): not enough input data!!");

  // データ行列の計算
    const u_int		ydim = first->second.dim(), xydim2 = xdim*ydim;
    Matrix<double>	M(xdim, xdim);
    Vector<double>	c(xdim), v(xydim2 + ydim);
    for (; first != last; ++first)
    {
	const Vector<double>&	x = first->first;
	const Vector<double>&	y = first->second;

	M += x % x;
	c += x;
	for (int j = 0; j < ydim; ++j)
	    v(j*xdim, xdim) += y[j]*x;
	v(xydim2, ydim) += y;
    }
    Matrix<double>	W(xydim2 + ydim, xydim2 + ydim);
    for (int j = 0; j < ydim; ++j)
    {
	W(j*xdim, j*xdim, xdim, xdim) = M;
	W[xydim2 + j](j*xdim, xdim)   = c;
	W[xydim2 + j][xydim2 + j]     = ndata;
    }
    W.symmetrize();

  // W*u = vを解いて変換パラメータを求める．
    v.solve(W);

  // 変換行列をセットする．
    _T.resize(ydim + 1, xdim + 1);
    _T(0, 0, ydim, xdim) = Matrix<double>((double*)v, ydim, xdim);
    for (int j = 0; j < ydim; ++j)
	 _T[j][xdim] = v[xydim2 + j];
    _T[ydim][xdim] = 1.0;
}

//! アフィン変換を求めるために必要な点対の最小個数を返す．
/*!
  現在設定されている入出力空間の次元をもとに計算される．
  \return	必要な点対の最小個数すなわち入力空間の次元mに対して m + 1
*/
inline u_int
AffineMapping::ndataMin() const
{
    return inDim() + 1;
}
    
}
#endif	/* !__TUMapping_h */
