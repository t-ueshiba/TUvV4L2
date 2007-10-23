/*
 *  $Id: Geometry++.h,v 1.20 2007-10-23 02:27:06 ueshiba Exp $
 */
#ifndef __TUGeometryPP_h
#define __TUGeometryPP_h

#include "TU/utility.h"
#include "TU/Minimize++.h"

namespace TU
{
/************************************************************************
*  class Point2<T>							*
************************************************************************/
template <class T>
class Point2 : public Vector<T, FixedSizedBuf<T, 2> >
{
  private:
    typedef Vector<T, FixedSizedBuf<T, 2> >	array_type;
    
  public:
    Point2(T u=0, T v=0)						;
    template <class T2, class B2>
    Point2(const Vector<T2, B2>& v) :array_type(v)			{}
    template <class T2, class B2>
    Point2&	operator =(const Vector<T2, B2>& v)
		{
		    array_type::operator =(v);
		    return *this;
		}
    Vector<T, FixedSizedBuf<T, 3> >
		hom()						const	;
    Point2	neighbor(int)					const	;
    Point2&	move(int)						;
    int		adj(const Point2&)				const	;
    int		dir(const Point2&)				const	;
    int		angle(const Point2&, const Point2&)		const	;
};

template <class T> inline
Point2<T>::Point2(T u, T v)
    :array_type()
{
    (*this)[0] = u;
    (*this)[1] = v;
}

template <class T> inline Point2<T>
Point2<T>::neighbor(int dir) const
{
    return Point2(*this).move(dir);
}

template <class T> inline Vector<T, FixedSizedBuf<T, 3> >
Point2<T>::hom() const
{
    Vector<T, FixedSizedBuf<T, 3> >	v;
    v[0] = (*this)[0];
    v[1] = (*this)[1];
    v[2] = 1;
    return v;
}

typedef Point2<short>					Point2s;
typedef Point2<int>					Point2i;
typedef Point2<float>					Point2f;
typedef Point2<double>					Point2d;

/************************************************************************
*  class Point3<T>							*
************************************************************************/
template <class T>
class Point3 : public Vector<T, FixedSizedBuf<T, 3> >
{
  private:
    typedef Vector<T, FixedSizedBuf<T, 3> >	array_type;
    
  public:
    Point3(T x=0, T y=0, T z=0)						;
    template <class T2, class B2>
    Point3(const Vector<T2, B2>& v) :array_type(v)			{}
    template <class T2, class B2>
    Point3&	operator =(const Vector<T2, B2>& v)
		{
		    array_type::operator =(v);
		    return *this;
		}
    Vector<T, FixedSizedBuf<T, 4> >
		hom()						const	;
};

template <class T> inline
Point3<T>::Point3(T x, T y, T z)
    :array_type()
{
    (*this)[0] = x;
    (*this)[1] = y;
    (*this)[2] = z;
}

template <class T> inline Vector<T, FixedSizedBuf<T, 4> >
Point3<T>::hom() const
{
    Vector<T, FixedSizedBuf<T, 4> >	v;
    v[0] = (*this)[0];
    v[1] = (*this)[1];
    v[2] = (*this)[2];
    v[3] = 1;
    return v;
}

typedef Point3<short>					Point3s;
typedef Point3<int>					Point3i;
typedef Point3<float>					Point3f;
typedef Point3<double>					Point3d;

/************************************************************************
*  class Normalize							*
************************************************************************/
//! 点の非同次座標の正規化変換を行うクラス
/*!
  \f$\TUud{x}{}=[\TUtvec{x}{}, 1]^\top~
  (\TUvec{x}{} \in \TUspace{R}{d})\f$に対して，以下のような平行移動と
  スケーリングを行う:
  \f[
	\TUud{y}{} =
	\TUbeginarray{c} s^{-1}(\TUvec{x}{} - \TUvec{c}{}) \\ 1	\TUendarray =
	\TUbeginarray{ccc}
	  s^{-1} \TUvec{I}{d} & -s^{-1}\TUvec{c}{} \\ \TUtvec{0}{d} & 1
	\TUendarray
	\TUbeginarray{c} \TUvec{x}{} \\ 1 \TUendarray =
	\TUvec{T}{}\TUud{x}{}
  \f]
  \f$s\f$と\f$\TUvec{c}{}\f$は，振幅の2乗平均値が空間の次元\f$d\f$に,
  重心が原点になるよう決定される．
*/
class Normalize
{
  public:
  //! 空間の次元を指定して正規化変換オブジェクトを生成する．
  /*!
    恒等変換として初期化される．
    \param d	空間の次元
  */
    Normalize(u_int d=2) :_npoints(0), _scale(1.0), _centroid(d)	{}

    template <class Iterator>
    Normalize(Iterator first, Iterator last)				;
    
    template <class Iterator>
    void		update(Iterator first, Iterator last)		;

    u_int		spaceDim()				const	;
    template <class T2, class B2>
    Vector<double>	operator ()(const Vector<T2, B2>& x)	const	;
    template <class T2, class B2>
    Vector<double>	normalizeP(const Vector<T2, B2>& x)	const	;
    
    Matrix<double>		T()				const	;
    Matrix<double>		Tt()				const	;
    Matrix<double>		Tinv()				const	;
    Matrix<double>		Ttinv()				const	;
    double			scale()				const	;
    const Vector<double>&	centroid()			const	;
    
  private:
    u_int		_npoints;	//!< これまでに与えた点の総数
    double		_scale;		//!< これまでに与えた点の振幅のRMS値
    Vector<double>	_centroid;	//!< これまでに与えた点群の重心
};

//! 与えられた点群の非同次座標から正規化変換オブジェクトを生成する．
/*!
  振幅の2乗平均値が#spaceDim(), 重心が原点になるような正規化変換が計算される．
  \param first	点群の先頭を示す反復子
  \param last	点群の末尾を示す反復子
*/
template <class Iterator> inline
Normalize::Normalize(Iterator first, Iterator last)
    :_npoints(0), _scale(1.0), _centroid()
{
    update(first, last);
}
    
//! 新たに点群を追加してその非同次座標から現在の正規化変換を更新する．
/*!
  振幅の2乗平均値が#spaceDim(), 重心が原点になるような正規化変換が計算される．
  \param first			点群の先頭を示す反復子
  \param last			点群の末尾を示す反復子
  \throw std::invalid_argument	これまでに与えられた点の総数が0の場合に送出
*/
template <class Iterator> void
Normalize::update(Iterator first, Iterator last)
{
    if (_npoints == 0)
    {
	if (first == last)
	    throw std::invalid_argument("Normalize::update(): 0-length input data!!");
	_centroid.resize(first->dim());
    }
    _scale = _npoints * (spaceDim() * _scale * _scale + _centroid * _centroid);
    _centroid *= _npoints;
    while (first != last)
    {
	_scale += first->square();
	_centroid += *first++;
	++_npoints;
    }
    if (_npoints == 0)
	throw std::invalid_argument("Normalize::update(): no input data accumulated!!");
    _centroid /= _npoints;
    _scale = sqrt((_scale / _npoints - _centroid * _centroid) / spaceDim());
}

//! この正規化変換が適用される空間の次元を返す．
/*! 
  \return	空間の次元(同次座標のベクトルとしての次元は#spaceDim()+1)
*/
inline u_int
Normalize::spaceDim() const
{
    return _centroid.dim();
}
    
//! 与えられた点に正規化変換を適用してその非同次座標を返す．
/*!
  \param x	点の非同次座標（#spaceDim()次元）
  \return	正規化された点の非同次座標（#spaceDim()次元）
*/
template <class T2, class B2> inline Vector<double>
Normalize::operator ()(const Vector<T2, B2>& x) const
{
    return (Vector<double>(x) -= _centroid) /= _scale;
}

//! 与えられた点に正規化変換を適用してその同次座標を返す．
/*!
  \param x	点の非同次座標（#spaceDim()次元）
  \return	正規化された点の同次座標（#spaceDim()+1次元）
*/
template <class T2, class B2> inline Vector<double>
Normalize::normalizeP(const Vector<T2, B2>& x) const
{
    Vector<double>	val(spaceDim()+1);
    val(0, spaceDim()) = (*this)(x);
    val[spaceDim()] = 1.0;
    return val;
}

//! 正規化変換のスケーリング定数を返す．
/*!
  \return	スケーリング定数（与えられた点列の振幅の2乗平均値）
*/
inline double
Normalize::scale() const
{
    return _scale;
}

//! 正規化変換の平行移動成分を返す．
/*!
  \return	平行移動成分（与えられた点列の重心）
*/
inline const Vector<double>&
Normalize::centroid() const
{
    return _centroid;
}

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
class ProjectiveMapping
{
  public:
    typedef double	ET;

  public:
    ProjectiveMapping(u_int inDim=2, u_int outDim=2)			;

  //! 変換行列を指定して射影変換オブジェクトを生成する．
  /*!
    \param T			(m+1)x(n+1)行列（m, nは入力／出力空間の次元）
  */
    ProjectiveMapping(const Matrix<double>& T)	:_T(T)			{}

    template <class Iterator>
    ProjectiveMapping(Iterator first, Iterator last, bool refine=false)	;

    template <class Iterator>
    void		initialize(Iterator first, Iterator last,
				   bool refine=false)			;

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
    Matrix<double>	jacobian(const Vector<S, B>& x)	const	;

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
    template <class T, class Iterator>
    class Cost
    {
      public:
	typedef double	ET;
	typedef T	AT;

	Cost(Iterator first, Iterator last)				;

	Vector<ET>	operator ()(const AT& map)		const	;
	Matrix<ET>	jacobian(const AT& map)			const	;
	static void	update(AT& map, const Vector<ET>& dm)		;
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
    initialize(first, last, refine);
}

//! 与えられた点対列の非同次座標から射影変換を計算する．
/*!
  \param first			点対列の先頭を示す反復子
  \param last			点対列の末尾を示す反復子
  \param refine			非線型最適化の有(true)／無(false)を指定
  \throw std::invalid_argument	点対の数が#ndataMin()に満たない場合に送出
*/
template <class Iterator> void
ProjectiveMapping::initialize(Iterator first, Iterator last, bool refine)
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
	throw std::invalid_argument("ProjectiveMapping::initialize(): not enough input data!!");

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
    const Vector<double>&	y = mapP(x);
    return y(0, outDim()) / y[outDim()];
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
    {
	Vector<double>	xx(inDim()+1);
	xx(0, inDim()) = x;
	xx[inDim()] = 1.0;
	return _T * xx;
    }
    else
	return _T * x;
}

//! 与えられた点におけるJacobianを返す．
/*!
  Jacobianとは射影変換行列成分に関する1階微分のことである．
  \param x	点の非同次座標（#inDim()次元）または同次座標（#inDim()+1次元）
  \return	Jacobian（#outDim() x (#outDim()+1)x(#inDim()+1)行列）
*/
template <class S, class B> Matrix<double>
ProjectiveMapping::jacobian(const Vector<S, B>& x) const
{
    Vector<double>		xP(inDim() + 1);
    if (x.dim() == inDim())
    {
	xP(0, inDim()) = x;
	xP[inDim()]    = 1.0;
    }
    else
	xP = x;
    const Vector<double>&	y = mapP(xP);
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
		（#outDim()+1次元）の対
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
		（#outDim()+1次元）の対
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
    
template <class T, class Iterator>
ProjectiveMapping::Cost<T, Iterator>::Cost(Iterator first, Iterator last)
    :_first(first), _last(last), _npoints(std::distance(_first, _last))
{
}
    
template <class T, class Iterator> Vector<double>
ProjectiveMapping::Cost<T, Iterator>::operator ()(const AT& map) const
{
    const u_int	outDim = map.outDim();
    Vector<ET>	val(_npoints*outDim);
    int	n = 0;
    for (Iterator iter = _first; iter != _last; ++iter)
    {
	val(n, outDim) = map(iter->first) - iter->second;
	n += outDim;
    }
    
    return val;
}
    
template <class T, class Iterator> Matrix<double>
ProjectiveMapping::Cost<T, Iterator>::jacobian(const AT& map) const
{
    const u_int	outDim = map.outDim();
    Matrix<ET>	J(_npoints*outDim, map.dof()+1);
    int	n = 0;
    for (Iterator iter = _first; iter != _last; ++iter)
    {
	J(n, 0, outDim, J.ncol()) = map.jacobian(iter->first);
	n += outDim;
    }

    return J;
}

template <class T, class Iterator> inline void
ProjectiveMapping::Cost<T, Iterator>::update(AT& map, const Vector<ET>& dm)
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
class AffineMapping : public ProjectiveMapping
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

    template <class Iterator>
    void	initialize(Iterator first, Iterator last)		;
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
    initialize(first, last);
}

//! 与えられた点対列の非同次座標からアフィン変換を計算する．
/*!
  \param first			点対列の先頭を示す反復子
  \param last			点対列の末尾を示す反復子
  \throw std::invalid_argument	点対の数が#ndataMin()に満たない場合に送出
*/
template <class Iterator> void
AffineMapping::initialize(Iterator first, Iterator last)
{
  // 充分な個数の点対があるか？
    const u_int		ndata = std::distance(first, last);
    if (ndata == 0)		// firstが有効か？
	throw std::invalid_argument("AffineMapping::initialize(): 0-length input data!!");
    const u_int		xdim = first->first.dim();
    if (ndata < xdim + 1)	// _Tのサイズが未定なのでndataMin()は無効
	throw std::invalid_argument("AffineMapping::initialize(): not enough input data!!");

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
    
/************************************************************************
*  class HyperPlane							*
************************************************************************/
//! d次元射影空間中の超平面を表現するクラス
/*!
  d次元射影空間の点\f$\TUud{x}{} \in \TUspace{R}{d+1}\f$に対して
  \f$\TUtud{p}{}\TUud{x}{} = 0,~\TUud{p}{} \in \TUspace{R}{d+1}\f$
  によって表される．
*/
template <class T, class B=Buf<T> >
class HyperPlane : public Vector<T, B>
{
  public:
    HyperPlane(u_int d=2)						;

  //! 同次座標ベクトルを指定して超平面オブジェクトを生成する．
  /*!
    \param p	(d+1)次元ベクトル（dは超平面が存在する射影空間の次元）
  */
    template <class T2, class B2>
    HyperPlane(const Vector<T2, B2>& p)	:Vector<T, B>(p)		{}

    template <class Iterator>
    HyperPlane(Iterator first, Iterator last)				;

  //! 超平面オブジェクトの同次座標ベクトルを指定する．
  /*!
    \param v	(d+1)次元ベクトル（dは超平面が存在する射影空間の次元）
    \return	この超平面オブジェクト
  */
    template <class T2, class B2>
    HyperPlane&	operator =(const Vector<T2, B2>& v)
				{Vector<T, B>::operator =(v); return *this;}

    template <class Iterator>
    void	fit(Iterator first, Iterator last)			;

  //! この超平面が存在する射影空間の次元を返す．
  /*! 
    \return	射影空間の次元(同次座標のベクトルとしての次元は#spaceDim()+1)
  */
    u_int	spaceDim()		const	{return Vector<T, B>::dim()-1;}

  //! 超平面を求めるために必要な点の最小個数を返す．
  /*!
    現在設定されている射影空間の次元をもとに計算される．
    \return	必要な点の最小個数すなわち入力空間の次元#spaceDim()
  */
    u_int	ndataMin()		const	{return spaceDim();}

    template <class T2, class B2> inline T
    sqdist(const Vector<T2, B2>& x)		const	;
    template <class T2, class B2> inline double
    dist(const Vector<T2, B2>& x)		const	;
};

//! 空間の次元を指定して超平面オブジェクトを生成する．
/*!
  無限遠超平面([0, 0,..., 0, 1])に初期化される．
  \param d	この超平面が存在する射影空間の次元
*/
template <class T, class B> inline
HyperPlane<T, B>::HyperPlane(u_int d)
    :Vector<T, B>(d + 1)
{
    (*this)[d] = 1;
}
    
//! 与えられた点列の非同次座標に当てはめられた超平面オブジェクトを生成する．
/*!
  \param first			点列の先頭を示す反復子
  \param last			点列の末尾を示す反復子
  \throw std::invalid_argument	点の数が#ndataMin()に満たない場合に送出
*/
template <class T, class B> template <class Iterator> inline
HyperPlane<T, B>::HyperPlane(Iterator first, Iterator last)
{
    fit(first, last);
}

//! 与えられた点列の非同次座標に超平面を当てはめる．
/*!
  \param first			点列の先頭を示す反復子
  \param last			点列の末尾を示す反復子
  \throw std::invalid_argument	点の数が#ndataMin()に満たない場合に送出
*/
template <class T, class B> template <class Iterator> void
HyperPlane<T, B>::fit(Iterator first, Iterator last)
{
  // 点列の正規化
    const Normalize	normalize(first, last);

  // 充分な個数の点があるか？
    const u_int		ndata = std::distance(first, last),
			d     = normalize.spaceDim();
    if (ndata < d)	// Vector<T, B>のサイズが未定なのでndataMin()は無効
	throw std::invalid_argument("Hyperplane::initialize(): not enough input data!!");

  // データ行列の計算
    Matrix<T>	A(d, d);
    while (first != last)
    {
	const Vector<T>&	x = normalize(*first++);
	A += x % x;
    }

  // データ行列の最小固有値に対応する固有ベクトルから法線ベクトルを計算し，
  // さらに点列の重心より原点からの距離を計算する．
    Vector<T>		eval;
    const Matrix<T>&	Ut = A.eigen(eval);
    Vector<T, B>::resize(d+1);
    (*this)(0, d) = Ut[Ut.nrow()-1];
    (*this)[d] = -((*this)(0, d)*normalize.centroid());
    if ((*this)[d] > 0.0)
	Vector<T, B>::operator *=(-1.0);
}

//! 与えられた点と超平面の距離の2乗を返す．
/*!
  \param x	点の非同次座標（#spaceDim()次元）または同次座標
		（#spaceDim()+1次元）
  \return	点と超平面の距離の2乗
*/
template <class T, class B> template <class T2, class B2> inline T
HyperPlane<T, B>::sqdist(const Vector<T2, B2>& x) const
{
    const double	d = dist(x);
    return d*d;
}

//! 与えられた点と超平面の距離を返す．
/*!
  \param x			点の非同次座標（#spaceDim()次元）または
				同次座標（#spaceDim()+1次元）
  \return			点と超平面の距離（非負）
  \throw std::invalid_argument	点のベクトルとしての次元が#spaceDim()，
				#spaceDim()+1のいずれでもない場合，もしくは
				この点が無限遠点である場合に送出．
*/
template <class T, class B> template <class T2, class B2> double
HyperPlane<T, B>::dist(const Vector<T2, B2>& x) const
{
    if (x.dim() == spaceDim())
    {
	Vector<T2>	xx(spaceDim()+1);
	xx(0, spaceDim()) = x;
	xx[spaceDim()] = 1;
	return fabs((*this * xx)/(*this)(0, spaceDim()).length());
    }
    else if (x.dim() == spaceDim() + 1)
    {
	if (x[spaceDim()] == 0.0)
	    throw std::invalid_argument("HyperPlane::dist(): point at infinitiy!!");
	return fabs(((*this) * x)/
		    ((*this)(0, spaceDim()).length() * x[spaceDim()]));
    }
    else
	throw std::invalid_argument("HyperPlane::dist(): dimension mismatch!!");

    return 0;
}

typedef HyperPlane<float,  FixedSizedBuf<float,  3> >	LineP2f;
typedef HyperPlane<double, FixedSizedBuf<double, 3> >	LineP2d;
typedef HyperPlane<float,  FixedSizedBuf<float,  4> >	PlaneP3f;
typedef HyperPlane<double, FixedSizedBuf<double, 4> >	PlaneP3d;

/************************************************************************
*  class CameraBase							*
************************************************************************/
//! すべての透視投影カメラの基底となるクラス
class CameraBase
{
  public:
  //! カメラの内部パラメータを表すクラス
    class Intrinsic
    {
      public:
	virtual ~Intrinsic()						;
	
      // various operations.
	virtual Point2d		operator ()(const Point2d& xc)	const	;
	virtual Point2d		xd(const Point2d& xc)		const	;
	virtual Matrix<double>	jacobianK(const Point2d& xc)	const	;
	virtual Matrix<double>	jacobianXC(const Point2d& xc)	const	;
	virtual Point2d		xc(const Point2d& u)		const	;

      // calibration matrices.    
	virtual Matrix<double>	K()				const	;
	virtual Matrix<double>	Kt()				const	;
	virtual Matrix<double>	Kinv()				const	;
	virtual Matrix<double>	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		k()				const	;
	virtual Point2d		principal()			const	;
	virtual double		aspect()			const	;
	virtual double		skew()				const	;
	virtual double		d1()				const	;
	virtual double		d2()				const	;
	virtual Intrinsic&	setFocalLength(double k)		;
	virtual Intrinsic&	setPrincipal(double u0, double v0)	;
	virtual Intrinsic&	setAspect(double aspect)		;
	virtual Intrinsic&	setSkew(double skew)			;
	virtual Intrinsic&	setIntrinsic(const Matrix<double>& K)	;
	virtual Intrinsic&	setDistortion(double d1, double d2)	;

      // parameter updating functions.
	virtual Intrinsic&	update(const Vector<double>& dp)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;
    };
    
  public:
  //! 位置を原点に，姿勢を単位行列にセットして初期化
    CameraBase()
	:_t(3), _Rt(3, 3)	{_Rt[0][0] = _Rt[1][1] = _Rt[2][2] = 1.0;}
  //! 位置と姿勢を単位行列にセットして初期化
  /*!
    \param t	カメラ位置を表す3次元ベクトル．
    \param Rt	カメラ姿勢を表す3x3回転行列．
  */
    CameraBase(const Vector<double>& t, const Matrix<double>& Rt)
	:_t(t), _Rt(Rt)							{}
    virtual ~CameraBase()						;
    
  // various operations in canonical coordinates.
    Point2d		xc(const Vector<double>& x)		const	;
    Point2d		xc(const Point2d& u)			const	;
    Matrix<double>	Pc()					const	;
    Matrix<double>	jacobianPc(const Vector<double>& x)	const	;
    Matrix<double>	jacobianXc(const Vector<double>& x)	const	;

  // various oeprations in image coordinates.
    Point2d		operator ()(const Vector<double>& x)	const	;
    Matrix<double>	P()					const	;
    Matrix<double>	jacobianP(const Vector<double>& x)	const	;
    Matrix<double>	jacobianFCC(const Vector<double>& x)	const	;
    Matrix<double>	jacobianX(const Vector<double>& x)	const	;
    Matrix<double>	jacobianK(const Vector<double>& x)	const	;
    Matrix<double>	jacobianXC(const Vector<double>& x)	const	;
    virtual CameraBase& setProjection(const Matrix<double>& P)		=0;

  // parameter updating functions.
    void		update(const Vector<double>& dp)		;
    void		updateFCC(const Vector<double>& dp)		;
    void		updateIntrinsic(const Vector<double>& dp)	;
    
  // calibration matrices.
    Matrix<double>	K()		const	{return intrinsic().K();}
    Matrix<double>	Kt()		const	{return intrinsic().Kt();}
    Matrix<double>	Kinv()		const	{return intrinsic().Kinv();}
    Matrix<double>	Ktinv()		const	{return intrinsic().Ktinv();}

  // extrinsic parameters.
    const Vector<double>&	t()	const	{return _t;}
    const Matrix<double>&	Rt()	const	{return _Rt;}
    CameraBase&		setTranslation(const Vector<double>& t)	;
    CameraBase&		setRotation(const Matrix<double>& Rt)	;

  // intrinsic parameters.
    virtual const Intrinsic&
			intrinsic()	const	= 0;
    virtual Intrinsic&	intrinsic()		= 0;
    u_int		dofIntrinsic()	const	{return intrinsic().dof();}
    double		k()		const	{return intrinsic().k();}
    Point2d		principal()	const	{return intrinsic().principal();}
    double		aspect()	const	{return intrinsic().aspect();}
    double		skew()		const	{return intrinsic().skew();}
    double		d1()		const	{return intrinsic().d1();}
    double		d2()		const	{return intrinsic().d2();}
    CameraBase&		setFocalLength(double k)		;
    CameraBase&		setPrincipal(double u0, double v0)	;
    CameraBase&		setAspect(double aspect)		;
    CameraBase&		setSkew(double skew)			;
    CameraBase&		setIntrinsic(const Matrix<double>& K)	;
    CameraBase&		setDistortion(double d1, double d2)	;
    
  // I/O functions.
    std::istream&	get(std::istream& in)			;
    std::ostream&	put(std::ostream& out)		const	;

  private:
    Vector<double>	_t;			// camera center.
    Matrix<double>	_Rt;			// camera orientation.
};

//! 3次元空間中の点の像のcanonicalカメラ座標系における位置を求める
/*!
  像は以下のように計算される．
  \f[
    \TUbeginarray{c} x_c \\ y_c \TUendarray = 
    \frac{1}{\TUtvec{r}{z}(\TUvec{x}{} - \TUvec{t}{})}
    \TUbeginarray{c}
      \TUtvec{r}{x}(\TUvec{x}{} - \TUvec{t}{}) \\
      \TUtvec{r}{y}(\TUvec{x}{} - \TUvec{t}{})
    \TUendarray
  \f]
  \param x	3次元空間中の点を表す3次元ベクトル．
  \return	xの像のcanonicalカメラ座標系における位置．
*/
inline Point2d
CameraBase::xc(const Vector<double>& x) const
{
    const Vector<double>&	xx = _Rt * (x - _t);
    return Point2d(xx[0] / xx[2], xx[1] / xx[2]);
}

//! 画像座標における点の2次元位置をcanonicalカメラ座標系に直す
/*!
  \param u	画像座標系における点の2次元位置．
  \return	canonicalカメラ座標系におけるuの2次元位置．
*/
inline Point2d
CameraBase::xc(const Point2d& u) const
{
    return intrinsic().xc(u);
}

//! 3次元空間中の点の像の画像座標系における位置を求める
/*!
  \param x	3次元空間中の点を表す3次元ベクトル．
  \return	xの像の画像座標系における位置．
*/
inline Point2d
CameraBase::operator ()(const Vector<double>& x) const
{
    return intrinsic()(xc(x));
}

//! 3次元ユークリッド空間から画像平面への投影行列を求める
/*!
  \return	投影行列．
*/
inline Matrix<double>
CameraBase::P() const
{
    return K() * Pc();
}

//! 位置を固定したときの内部/外部パラメータに関するJacobianを求める
/*!
  \return	
*/
inline Matrix<double>
CameraBase::jacobianFCC(const Vector<double>& x) const
{
    const Matrix<double>&	J = jacobianP(x);
    return Matrix<double>(J, 0, 3, J.nrow(), J.ncol() - 3);
}

inline Matrix<double>
CameraBase::jacobianX(const Vector<double>& x) const
{
    return intrinsic().jacobianXC(xc(x)) * jacobianXc(x);
}

inline Matrix<double>
CameraBase::jacobianK(const Vector<double>& x) const
{
    return intrinsic().jacobianK(xc(x));
}

inline Matrix<double>
CameraBase::jacobianXC(const Vector<double>& x) const
{
    return intrinsic().jacobianXC(xc(x));
}

inline void
CameraBase::updateIntrinsic(const Vector<double>& dp)
{
    intrinsic().update(dp);			// update intrinsic parameters.
}

inline void
CameraBase::updateFCC(const Vector<double>& dp)
{
    _Rt *= Matrix<double>::Rt(dp(0, 3));	// update rotation.
    updateIntrinsic(dp(3, dp.dim() - 3));	// update intrinsic parameters.
}

inline void
CameraBase::update(const Vector<double>& dp)
{
    _t -= dp(0, 3);				// update translation.
    updateFCC(dp(3, dp.dim() - 3));		// update other prameters.
}

inline CameraBase&
CameraBase::setTranslation(const Vector<double>& t)
{
    _t = t;
    return *this;
}

inline CameraBase&
CameraBase::setRotation(const Matrix<double>& Rt)
{
    _Rt = Rt;
    return *this;
}

inline CameraBase&
CameraBase::setFocalLength(double k)
{
    intrinsic().setFocalLength(k);
    return *this;
}

inline CameraBase&
CameraBase::setPrincipal(double u0, double v0)
{
    intrinsic().setPrincipal(u0, v0);
    return *this;
}

inline CameraBase&
CameraBase::setAspect(double aspect)
{
    intrinsic().setAspect(aspect);
    return *this;
}

inline CameraBase&
CameraBase::setSkew(double skew)
{
    intrinsic().setSkew(skew);
    return *this;
}

inline CameraBase&
CameraBase::setIntrinsic(const Matrix<double>& K)
{
    intrinsic().setIntrinsic(K);
    return *this;
}

inline CameraBase&
CameraBase::setDistortion(double d1, double d2)
{
    intrinsic().setDistortion(d1, d2);
    return *this;
}

inline std::istream&
operator >>(std::istream& in, CameraBase& camera)
{
    return camera.get(in);
}

inline std::ostream&
operator <<(std::ostream& out, const CameraBase& camera)
{
    return camera.put(out);
}

inline std::istream&
operator >>(std::istream& in, CameraBase::Intrinsic& intrinsic)
{
    return intrinsic.get(in);
}

inline std::ostream&
operator <<(std::ostream& out, const CameraBase::Intrinsic& intrinsic)
{
    return intrinsic.put(out);
}

/************************************************************************
*  class CanonicalCamera						*
************************************************************************/
class CanonicalCamera : public CameraBase
{
  public:
    CanonicalCamera()	:CameraBase(), _intrinsic()	{}
    CanonicalCamera(const Vector<double>& t, const Matrix<double>& Rt)
	:CameraBase(t, Rt), _intrinsic()		{}
    CanonicalCamera(const Matrix<double>& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&	setProjection(const Matrix<double>& P)		;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;

  private:
    Intrinsic	_intrinsic;
};

/************************************************************************
*  class CameraWithFocalLength						*
************************************************************************/
class CameraWithFocalLength : public CameraBase
{
  public:
    class Intrinsic : public CanonicalCamera::Intrinsic
    {
      public:
	Intrinsic(double k=1.0)	:_k(k)					{}

      // various operations.
	virtual Point2d		operator ()(const Point2d& xc)	const	;
	virtual Matrix<double>	jacobianK(const Point2d& xc)	const	;
	virtual Matrix<double>	jacobianXC(const Point2d& xc)	const	;
	virtual Point2d		xc(const Point2d& u)		const	;

      // calibration matrices.
	virtual Matrix<double>	K()				const	;
	virtual Matrix<double>	Kt()				const	;
	virtual Matrix<double>	Kinv()				const	;
	virtual Matrix<double>	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		k()				const	;
	virtual	CameraBase::Intrinsic&
				setFocalLength(double k)		;

      // parameter updating functions.
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;

      private:
	double	_k;
    };
    
  public:
    CameraWithFocalLength()	:CameraBase(), _intrinsic()	{}
    CameraWithFocalLength(const Vector<double>& t,
			  const Matrix<double>& Rt,
			  double		k=1.0)
	:CameraBase(t, Rt), _intrinsic(k)			{}
    CameraWithFocalLength(const Matrix<double>& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&		setProjection(const Matrix<double>& P)	;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};

/************************************************************************
*  class CameraWithEuclideanImagePlane					*
************************************************************************/
class CameraWithEuclideanImagePlane : public CameraBase
{
  public:
    class Intrinsic : public CameraWithFocalLength::Intrinsic
    {
      public:
	Intrinsic(double k=1.0, double u0=0.0, double v0=0.0)
	    :CameraWithFocalLength::Intrinsic(k), _principal(u0, v0)	{}
	Intrinsic(const CameraWithFocalLength::Intrinsic& intrinsic)
	    :CameraWithFocalLength::Intrinsic(intrinsic),
	     _principal(0.0, 0.0)					{}
	
      // various operations.	
	virtual Point2d		operator ()(const Point2d& xc)	const	;
	virtual Matrix<double>	jacobianK(const Point2d& xc)	const	;
	virtual Point2d		xc(const Point2d& u)		const	;
    
      // calibration matrices.	
	virtual Matrix<double>	K()				const	;
	virtual Matrix<double>	Kt()				const	;
	virtual Matrix<double>	Kinv()				const	;
	virtual Matrix<double>	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual Point2d		principal()			const	;
	virtual CameraBase::Intrinsic&
				setPrincipal(double u0, double v0)	;

      // parameter updating functions.
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;

      private:
	Point2d	_principal;
    };
    
  public:
    CameraWithEuclideanImagePlane()	:CameraBase(), _intrinsic()	{}
    CameraWithEuclideanImagePlane(const Vector<double>& t,
				  const Matrix<double>& Rt,
				  double		k=1.0,
				  double		u0=0,
				  double		v0=0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0)			{}
    CameraWithEuclideanImagePlane(const Matrix<double>& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&	setProjection(const Matrix<double>& P)		;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};
    
/************************************************************************
*  class Camera								*
************************************************************************/
class Camera : public CameraBase
{
  public:
    class Intrinsic : public CameraWithEuclideanImagePlane::Intrinsic
    {
      public:
	Intrinsic(double k=1.0, double u0=0.0, double v0=0.0,
		  double aspect=1.0, double skew=0.0)
	    :CameraWithEuclideanImagePlane::Intrinsic(k, u0, v0),
	     _k00(aspect * k), _k01(skew * k)				{}
	Intrinsic(const CameraWithEuclideanImagePlane::Intrinsic& intrinsic)
	    :CameraWithEuclideanImagePlane::Intrinsic(intrinsic),
	     _k00(k()), _k01(0.0)					{}
	Intrinsic(const Matrix<double>& K)
	    :CameraWithEuclideanImagePlane::Intrinsic(),
	     _k00(k()), _k01(0.0)			{setIntrinsic(K);}
	
      // various operations.
	virtual Point2d		operator ()(const Point2d& xc)	const	;
	virtual Matrix<double>	jacobianK(const Point2d& xc)	const	;
	virtual Matrix<double>	jacobianXC(const Point2d& xc)	const	;
	virtual Point2d		xc(const Point2d& u)		const	;

      // calibration matrices.
	virtual Matrix<double>	K()				const	;
	virtual Matrix<double>	Kt()				const	;
	virtual Matrix<double>	Kinv()				const	;
	virtual Matrix<double>	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		aspect()			const	;
	virtual double		skew()				const	;
	virtual	CameraBase::Intrinsic&
				setFocalLength(double k)		;
	virtual CameraBase::Intrinsic&
				setAspect(double aspect)		;
	virtual CameraBase::Intrinsic&
				setSkew(double skew)			;
	virtual CameraBase::Intrinsic&
				setIntrinsic(const Matrix<double>& K)	;

      // parameter updating functions.
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;
    
      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;

      protected:
		double		k00()			const	{return _k00;}
		double		k01()			const	{return _k01;}
	
      private:
	double	_k00, _k01;
    };
    
  public:
    Camera()	:CameraBase(), _intrinsic()			{}
    Camera(const Vector<double>& t,
	   const Matrix<double>& Rt,
	   double		 k=1.0,
	   double		 u0=0,
	   double		 v0=0,
	   double		 aspect=1.0,
	   double		 skew=0.0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0, aspect, skew)	{}
    Camera(const Matrix<double>& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&	setProjection(const Matrix<double>& P);
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};

/************************************************************************
*  class CameraWithDistortion						*
************************************************************************/
class CameraWithDistortion : public CameraBase
{
  public:
    class Intrinsic : public Camera::Intrinsic
    {
      public:
	Intrinsic(double k=1.0, double u0=0.0, double v0=0.0,
		  double aspect=1.0, double skew=0.0,
		  double d1=0.0, double d2=0.0)
	    :Camera::Intrinsic(k, u0, v0, aspect, skew),
	     _d1(d1), _d2(d2)						{}
	Intrinsic(const Camera::Intrinsic& intrinsic)
	    :Camera::Intrinsic(intrinsic), _d1(0.0), _d2(0.0)		{}
	Intrinsic(const Matrix<double>& K)
	    :Camera::Intrinsic(), _d1(0.0), _d2(0.0)	{setIntrinsic(K);}
	
      // various operations.
	virtual Point2d		operator ()(const Point2d& xc)	const	;
	virtual Point2d		xd(const Point2d& xc)		const	;
	virtual Matrix<double>	jacobianXC(const Point2d& xc)	const	;
	virtual Matrix<double>	jacobianK(const Point2d& xc)	const	;
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;
	virtual Point2d		xc(const Point2d& u)		const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		d1()				const	;
	virtual double		d2()				const	;
	virtual CameraBase::Intrinsic&	
				setDistortion(double d1, double d2)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;

      private:
	double	_d1, _d2;
    };
    
  public:
    CameraWithDistortion()	:CameraBase(), _intrinsic()		{}
    CameraWithDistortion(const Vector<double>& t,
			 const Matrix<double>& Rt,
			 double		       k=1.0,
			 double		       u0=0,
			 double		       v0=0,
			 double		       aspect=1.0,
			 double		       skew=0.0,
			 double		       d1=0.0,
			 double		       d2=0.0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0, aspect, skew, d1, d2)	{}
    CameraWithDistortion(const Matrix<double>& P,
			 double d1=0.0, double d2=0.0)			;

    virtual CameraBase&		setProjection(const Matrix<double>& P)	;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};
 
inline
CameraWithDistortion::CameraWithDistortion(const Matrix<double>& P,
					   double d1, double d2)
    :CameraBase(), _intrinsic()
{
    setProjection(P);
    setDistortion(d1,d2);
}

}
#endif	/* !__TUGeometryPP_h */
