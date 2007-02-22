/*
 *  $Id: Geometry++.h,v 1.13 2007-02-22 23:23:21 ueshiba Exp $
 */
#ifndef __TUGeometryPP_h
#define __TUGeometryPP_h

#include "TU/Minimize++.h"

namespace TU
{
/************************************************************************
*  class CoordBase<T, D>						*
************************************************************************/
template <class T, u_int D>
class CoordBase
{
  public:
    typedef T	ET;
    
    CoordBase()							{*this = 0.0;}
    CoordBase(const CoordBase&)					;
    CoordBase(const Vector<double>&)				;
    CoordBase&		operator =(const CoordBase&)		;
    CoordBase&		operator =(const Vector<double>&)	;
    
    CoordBase&		operator +=(const CoordBase&)		;
    CoordBase&		operator -=(const CoordBase&)		;
    CoordBase&		operator  =(double c)			;
    CoordBase&		operator *=(double c)			;
    CoordBase&		operator /=(double c)			;
    CoordBase		operator  -()			const	{CoordBase
								     r(*this);
								 r *= -1;
								 return r;}
			operator T*()			const	{return(T*)_p;}
			operator Vector<double>()	const	;
    static u_int	dim()					{return D;}
    const T&		operator [](int i)		const	{return _p[i];}
    T&			operator [](int i)			{return _p[i];}
    int			operator ==(const CoordBase&)	const	;
    int			operator !=(const CoordBase& p) const
			{
			    return !(*this == p);
			}
    std::istream&	restore(std::istream&)		;
    std::ostream&	save(std::ostream&)	const	;
    double		square()		const	;
    double		length()		const	{return
							   sqrt(square());}
    CoordBase&		normalize()			{return
							   *this /= length();}
    CoordBase		normal()		const	;
    void		check_dim(u_int d)	const	;

  private:
    T		_p[D];
};

template <class T, u_int D> inline double
CoordBase<T, D>::square() const
{
    return Vector<double>(*this).square();
}

/*
 *  I/O functions
 */
template <class T, u_int D> std::istream&
operator >>(std::istream& in, CoordBase<T, D>& p)		;
template <class T, u_int D> std::ostream&
operator <<(std::ostream& out, const CoordBase<T, D>& p)	;

/*
 *  numerical operators
 */
template <class T, u_int D> T
operator *(const CoordBase<T, D>&, const CoordBase<T, D>&)	;

template <class T, u_int D> inline CoordBase<T, D>
operator +(const CoordBase<T, D>& a, const CoordBase<T, D>& b)
    {CoordBase<T, D> r(a); r += b; return r;}

template <class T, u_int D> inline CoordBase<T, D>
operator -(const CoordBase<T, D>& a, const CoordBase<T, D>& b)
    {CoordBase<T, D> r(a); r -= b; return r;}

template <class T, u_int D> inline CoordBase<T, D>
operator *(double c, const CoordBase<T, D>& a)
    {CoordBase<T, D> r(a); r *= c; return r;}

template <class T, u_int D> inline CoordBase<T, D>
operator *(const Array<T>& a, double c)
    {CoordBase<T, D> r(a); r *= c; return r;}

template <class T, u_int D> inline CoordBase<T, D>
operator /(const Array<T>& a, double c)
    {CoordBase<T, D> r(a); r /= c; return r;}

/************************************************************************
*  class Coordinate<T, D>						*
************************************************************************/
template <class T, u_int D>
class CoordinateP;

template <class T, u_int D>
class Coordinate : public CoordBase<T, D>
{
  public:
    Coordinate()				:CoordBase<T, D>()	{}
    Coordinate(const Coordinate& p)		:CoordBase<T, D>(p)	{}
    Coordinate(const Vector<double>& v)	:CoordBase<T, D>(v)		{}
    Coordinate(const CoordinateP<T, D>&)				;
    Coordinate&		operator =(const Coordinate& p)
			{CoordBase<T, D>::operator =(p); return *this;}
    Coordinate&		operator =(const Vector<double>& v)
			{CoordBase<T, D>::operator =(v); return *this;}
    Coordinate&		operator =(const CoordinateP<T, D>&)		;

    using		CoordBase<T, D>::dim;
    Coordinate&		operator +=(const Coordinate& p)
			{
			    CoordBase<T, D>::operator +=(p);
			    return *this;
			}
    Coordinate&		operator -=(const Coordinate& p)
			{
			    CoordBase<T, D>::operator -=(p);
			    return *this;
			}
    Coordinate&		operator  =(double c)
			{
			    CoordBase<T, D>::operator =(c);
			    return *this;
			}
    Coordinate&		operator *=(double c)
			{
			    CoordBase<T, D>::operator *=(c);
			    return *this;
			}
    Coordinate&		operator /=(double c)
			{
			    CoordBase<T, D>::operator /=(c);
			    return *this;
			}
    Coordinate		operator -() const
			{
			    Coordinate	r(*this);
			    r *= -1;
			    return r;
			}
    double		sqdist(const Coordinate& p)		 const	;
    double		dist(const Coordinate& p) const
			{
			    return sqrt(sqdist(p));
			}
};

/*
 *  numerical operators
 */
template <class T> Coordinate<T, 3u>
operator ^(const Coordinate<T, 3u>&, const Coordinate<T, 3u>&);

template <class T, u_int D> inline Coordinate<T, D>
operator +(const Coordinate<T, D>& a, const Coordinate<T, D>& b)
    {Coordinate<T, D> r(a); r += b; return r;}

template <class T, u_int D> inline Coordinate<T, D>
operator -(const Coordinate<T, D>& a, const Coordinate<T, D>& b)
    {Coordinate<T, D> r(a); r -= b; return r;}

template <class T, u_int D> inline Coordinate<T, D>
operator *(double c, const Coordinate<T, D>& a)
    {Coordinate<T, D> r(a); r *= c; return r;}

template <class T, u_int D> inline Coordinate<T, D>
operator *(const Coordinate<T, D>& a, double c)
    {Coordinate<T, D> r(a); r *= c; return r;}

template <class T, u_int D> inline Coordinate<T, D>
operator /(const Coordinate<T, D>& a, double c)
    {Coordinate<T, D> r(a); r /= c; return r;}

/************************************************************************
*  class CoordinateP<T, D>						*
************************************************************************/
template <class T, u_int D>
class CoordinateP : public CoordBase<T, D+1u>
{
  public:
    CoordinateP()			     :CoordBase<T, D+1u>()	{}
    CoordinateP(const CoordinateP& p)    :CoordBase<T, D+1u>(p)	{}
    CoordinateP(const Vector<double>& v) :CoordBase<T, D+1u>(v)	{}
    CoordinateP(const Coordinate<T, D>&)				;

    using		CoordBase<T, D+1u>::dim;
    CoordinateP&	operator =(const CoordinateP& p)
			{CoordBase<T, D+1u>::operator =(p); return *this;}
    CoordinateP&	operator =(const Vector<double>& v)
			{CoordBase<T, D+1u>::operator =(v); return *this;}
    CoordinateP&	operator =(const Coordinate<T, D>&)		;

    CoordinateP&	operator  =(double c)
			{
			    CoordBase<T, D+1u>::operator =(c);
			    return *this;
			}
    CoordinateP&	operator *=(double c)
			{
			    CoordBase<T, D+1u>::operator *=(c);
			    return *this;
			}
    CoordinateP&	operator /=(double c)
			{
			    CoordBase<T, D+1u>::operator /=(c);
			    return *this;
			}
    CoordinateP		operator -() const
			{
			    CoordinateP	r(*this);
			    r *= -1;
			    return r;
			}
    int			operator ==(const CoordinateP& p)	const	;
};

/*
 *  numerical operators
 */
template <class T> CoordinateP<T, 2u>
operator ^(const CoordinateP<T, 2u>&, const CoordinateP<T, 2u>&);

template <class T, u_int D> inline CoordinateP<T, D>
operator *(double c, const CoordinateP<T, D>& a)
    {CoordinateP<T, D> r(a); r *= c; return r;}

template <class T, u_int D> inline CoordinateP<T, D>
operator *(const CoordinateP<T, D>& a, double c)
    {CoordinateP<T, D> r(a); r *= c; return r;}

template <class T, u_int D> inline CoordinateP<T, D>
operator /(const CoordinateP<T, D>& a, double c)
    {CoordinateP<T, D> r(a); r /= c; return r;}

/************************************************************************
*  class Point2<T>							*
************************************************************************/
template <class T>	class PointP2;
template <class T>
class Point2 : public Coordinate<T, 2u>
{
  public:
    Point2()				:Coordinate<T, 2u>()	{}
    Point2(T, T)						;
    Point2(const Point2<T>& p)
	:Coordinate<T, 2u>((const Coordinate<T, 2u>&)p)	{}
    Point2(const PointP2<T>& p)
	:Coordinate<T, 2u>((const CoordinateP<T, 2u>&)p)	{}
    Point2(const Vector<double>& v)	:Coordinate<T, 2u>(v)	{}
    Point2&	operator =(const Point2<T>& p)
		{
		    Coordinate<T, 2u>::operator =(p);
		    return *this;
		}
    Point2&	operator =(const PointP2<T>& p)
		{
		    Coordinate<T, 2u>::operator =(p);
		    return *this;
		}
    Point2&	operator =(const Vector<double>& v)
		{
		    Coordinate<T, 2u>::operator =(v);
		    return *this;
		}
    Point2&	operator +=(const Point2& p)
		{
		    Coordinate<T, 2u>::operator +=(p);
		    return *this;
		}
    Point2&	operator -=(const Point2& p)
		{
		    Coordinate<T, 2u>::operator -=(p);
		    return *this;
		}
    Point2&	operator  =(double c)
		{
		    Coordinate<T, 2u>::operator =(c);
		    return *this;
		}
    Point2&	operator *=(double c)
		{
		    Coordinate<T, 2u>::operator *=(c);
		    return *this;
		}
    Point2&	operator /=(double c)
		{
		    Coordinate<T, 2u>::operator /=(c);
		    return *this;
		}
    Point2	operator -() const
		{
		    Point2	r(*this);
		    r *= -1;
		    return r;
		}

    Point2	neighbor(int)				const	;
    Point2&	move(int)					;
    int		adj(const Point2&)			const	;
    int		dir(const Point2&)			const	;
    int		angle(const Point2&, const Point2&)	const	;
};

template <class T> inline
Point2<T>::Point2(T u, T v)
    :Coordinate<T, 2u>()
{
    (*this)[0] = u;
    (*this)[1] = v;
}

template <class T> inline Point2<T>
Point2<T>::neighbor(int dir) const
{
    Point2<T>	d(*this);
    return d.move(dir);
}

/************************************************************************
*  class PointP2<T>							*
************************************************************************/
template <class T>
class PointP2 : public CoordinateP<T, 2u>
{
  public:
    PointP2()				:CoordinateP<T, 2u>()		{}
    PointP2(T, T)							;
    PointP2(const PointP2<T>& p)
	:CoordinateP<T, 2u>((const CoordinateP<T, 2u>&)p)		{}
    PointP2(const Point2<T>& p)
	:CoordinateP<T, 2u>((const Coordinate<T, 2u>&)p)		{}
    PointP2(const Vector<double>& v)	:CoordinateP<T, 2u>(v)		{}
    PointP2&	operator =(const PointP2<T>& p)
		{
		    CoordinateP<T, 2u>::operator =(p);
		    return *this;
		}
    PointP2&	operator =(const Point2<T>& p)
		{
		    CoordinateP<T, 2u>::operator =(p);
		    return *this;
		}
    PointP2&	operator =(const Vector<double>& v)
		{
		    CoordinateP<T, 2u>::operator =(v);
		    return *this;
		}
    PointP2&	operator  =(double c)
		{
		    CoordinateP<T, 2u>::operator =(c);
		    return *this;
		}
    PointP2&	operator *=(double c)
		{
		    CoordinateP<T, 2u>::operator *=(c);
		    return *this;
		}
    PointP2&	operator /=(double c)
		{
		    CoordinateP<T, 2u>::operator /=(c);
		    return *this;
		}
    PointP2	operator -() const
		{
		    PointP2	r(*this);
		    r *= -1;
		    return r;
		}
    PointP2	neighbor(int)					const	;
    PointP2&	move(int)						;
    int		adj(const PointP2&)				const	;
    int		dir(const PointP2&)				const	;
    int		angle(const PointP2&, const PointP2&)		const	;
};

template <class T> inline
PointP2<T>::PointP2(T u, T v)
    :CoordinateP<T, 2u>()
{
    (*this)[0] = u;
    (*this)[1] = v;
    (*this)[2] = 1;
}

template <class T> inline PointP2<T>
PointP2<T>::neighbor(int dir) const
{
    Point2<T>	d(*this);
    return d.move(dir);
}

template <class T> inline PointP2<T>&
PointP2<T>::move(int dir)
{
    Point2<T>	d(*this);
    *this = d.move(dir);
    return *this;
}

template <class T> int
PointP2<T>::dir(const PointP2<T>& p) const
{
    Point2<T>	d(*this);
    return d.dir(p);
}

template <class T> int
PointP2<T>::angle(const PointP2<T>& pp, const PointP2<T>& pn) const
{
    Point2<T>	d(*this);
    return d.angle(pp, pn);
}

/************************************************************************
*  class LineP2<T>							*
************************************************************************/
template <class T>
class LineP2 : public CoordinateP<T, 2u>
{
  public:
    LineP2()				:CoordinateP<T, 2u>()		{}
    LineP2(const LineP2<T>& l)
	:CoordinateP<T, 2u>((const CoordinateP<T, 2u>&)l)		{}
    LineP2(const Vector<double>& v)	:CoordinateP<T, 2u>(v)		{}

    CoordinateP<T, 2u>::operator [];	// should not be needed, but due to
					// the bug in SGI C++ compiler....
};

/************************************************************************
*  class PlaneP3<T>							*
************************************************************************/
template <class T>
class PlaneP3 : public CoordinateP<T, 3u>
{
  public:
    PlaneP3()				:CoordinateP<T, 3u>()		{}
    PlaneP3(const PlaneP3<T>& l)
	:CoordinateP<T, 3u>((const CoordinateP<T, 3u>&)l)		{}
    PlaneP3(const Vector<double>& v):CoordinateP<T, 3u>(v)		{}
};

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
    Vector<double>	operator ()(const Vector<double>& x)	const	;
    Vector<double>	normalizeP(const Vector<double>& x)	const	;
    
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
inline Vector<double>
Normalize::operator ()(const Vector<double>& x) const
{
    return (x - _centroid) / _scale;
}

//! 与えられた点に正規化変換を適用してその同次座標を返す．
/*!
  \param x	点の非同次座標（#spaceDim()次元）
  \return	正規化された点の同次座標（#spaceDim()+1次元）
*/
inline Vector<double>
Normalize::normalizeP(const Vector<double>& x) const
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
    
    Vector<double>	operator ()(const Vector<double>& x)	const	;
    Vector<double>	mapP(const Vector<double>& x)		const	;

    template <class In, class Out>
    double		sqdist(const std::pair<In, Out>& pair)	const	;
    template <class In, class Out>
    double		dist(const std::pair<In, Out>& pair)	const	;
    
  protected:
    Matrix<double>	_T;			//!< 射影変換を表現する行列

  private:
  //! 射影変換行列の最尤推定のためのコスト関数
    class Cost
    {
      public:
	typedef double		ET;
	typedef Matrix<ET>	AT;

	template <class Iterator>
	Cost(Iterator first, Iterator last)				;

	Vector<ET>	operator ()(const AT& T)		const	;
	Matrix<ET>	jacobian(const AT& T)			const	;
	void		update(AT& T, const Vector<ET>& dt)	const	;

	u_int		npoints()		const	{return _X.nrow();}
	     
      private:
	Matrix<ET>	_X, _Y;
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
    const u_int		ndata = std::distance(first, last),
			xdim  = xNormalize.spaceDim() + 1,
			ydim  = yNormalize.spaceDim() + 1;
    if (ndata*(ydim - 1) < xdim*ydim - 1)	// _Tのサイズが未定なので
						// ndataMin()は無効
	throw std::invalid_argument("ProjectiveMapping::initialize(): not enough input data!!");

  // データ行列の計算
    Matrix<double>	A(xdim*ydim + ndata, xdim*ydim + ndata);
    int			n = xdim*ydim;
    for (Iterator iter = first; iter != last; ++iter)
    {
	const Vector<double>&	x = xNormalize.normalizeP(iter->first);
	const Vector<double>&	y = yNormalize.normalizeP(iter->second);

	A(0, 0, xdim, xdim) += x % x;
	for (int j = 0; j < ydim; ++j)
	    A[n](j*xdim, xdim) = -y[j] * x;
	A[n][n] = y * y;
	++n;
    }
    for (int j = 1; j < ydim; ++j)
	A(j*xdim, j*xdim, xdim, xdim) = A(0, 0, xdim, xdim);
    A.symmetrize();

  // データ行列の最小固有値に対応する固有ベクトルから変換行列を計算し，
  // 正規化をキャンセルする．
    Vector<double>		eval;
    const Matrix<double>&	Ut = A.eigen(eval);
    _T = yNormalize.Tinv()
       * Matrix<double>((double*)Ut[Ut.nrow()-1], ydim, xdim)
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
	Cost					cost(first, last);
	ConstNormConstraint<Matrix<double> >	constraint(_T);
	minimizeSquare(cost, constraint, _T);
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
inline Vector<double>
ProjectiveMapping::operator ()(const Vector<double>& x)	const
{
    const Vector<double>&	y = mapP(x);
    return y(0, outDim()) / y[outDim()];
}

//! 与えられた点に射影変換を適用してその同次座標を返す．
/*!
  \param x	点の非同次座標（#inDim()次元）または同次座標（#inDim()+1次元）
  \return	射影変換された点の同次座標（#outDim()+1次元）
*/
inline Vector<double>
ProjectiveMapping::mapP(const Vector<double>& x) const
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
    
template <class Iterator>
ProjectiveMapping::Cost::Cost(Iterator first, Iterator last)
    :_X(), _Y()
{
    const u_int	ndata = std::distance(first, last);
    _X.resize(ndata, first->first.dim() + 1);
    _Y.resize(ndata, first->second.dim());
    for (int n = 0; first != last; ++first)
    {
	_X[n](0, _X.ncol() - 1) = first->first;
	_X[n][_X.ncol() - 1]	= 1.0;
	_Y[n++]			= first->second;
    }
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
    const u_int	ndata = std::distance(first, last);
    if (ndata == 0)		// firstが有効か？
	throw std::invalid_argument("AffineMapping::initialize(): 0-length input data!!");
    const u_int	xdim = first->first.dim();
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
class HyperPlane
{
  public:
    HyperPlane(u_int d=2)						;

  //! 同次座標ベクトルを指定して超平面オブジェクトを生成する．
  /*!
    \param p	(d+1)次元ベクトル（dはもとの射影区間の次元）
  */
    HyperPlane(const Vector<double>& p)	:_p(p)				{}

    template <class Iterator>
    HyperPlane(Iterator first, Iterator last)				;

    template <class Iterator>
    void	initialize(Iterator first, Iterator last)		;

  //! この超平面が存在する射影空間の次元を返す．
  /*! 
    \return	射影空間の次元(同次座標のベクトルとしての次元は#spaceDim()+1)
  */
    u_int	spaceDim()			const	{return _p.dim()-1;}

  //! 超平面を求めるために必要な点の最小個数を返す．
  /*!
    現在設定されている射影空間の次元をもとに計算される．
    \return	必要な点の最小個数すなわち入力空間の次元#spaceDim()
  */
    u_int	ndataMin()			const	{return spaceDim();}

  //! この超平面を表現する同次座標ベクトルを返す．
  /*!
    \return	(#spaceDim()+1)次元ベクトル
  */
    const Vector<double>&
		p()				const	{return _p;}

    double	sqdist(const Vector<double>& x)	const	;
    double	dist(const Vector<double>& x)	const	;
    
  private:
    Vector<double>	_p;	//!> 超平面を表現するベクトル
};

//! 空間の次元を指定して超平面オブジェクトを生成する．
/*!
  無限遠超平面([0, 0,..., 0, 1])に初期化される．
  \param d	この超平面が存在する射影空間の次元
*/
inline
HyperPlane::HyperPlane(u_int d)
    :_p(d + 1)
{
    _p[d] = 1.0;
}
    
//! 与えられた点列の非同次座標に当てはめられた超平面オブジェクトを生成する．
/*!
  \param first			点列の先頭を示す反復子
  \param last			点列の末尾を示す反復子
  \throw std::invalid_argument	点の数が#ndataMin()に満たない場合に送出
*/
template <class Iterator> inline
HyperPlane::HyperPlane(Iterator first, Iterator last)
{
    initialize(first, last);
}

//! 与えられた点列の非同次座標に超平面を当てはめる．
/*!
  \param first			点列の先頭を示す反復子
  \param last			点列の末尾を示す反復子
  \throw std::invalid_argument	点の数が#ndataMin()に満たない場合に送出
*/
template <class Iterator> void
HyperPlane::initialize(Iterator first, Iterator last)
{
  // 点列の正規化
    const Normalize	normalize(first, last);

  // 充分な個数の点があるか？
    const u_int		ndata = std::distance(first, last),
			d     = normalize.spaceDim();
    if (ndata < d)	// _pのサイズが未定なのでndataMin()は無効
	throw std::invalid_argument("Hyperplane::initialize(): not enough input data!!");

  // データ行列の計算
    Matrix<double>	A(d, d);
    while (first != last)
    {
	const Vector<double>&	x = normalize(*first++);
	A += x % x;
    }

  // データ行列の最小固有値に対応する固有ベクトルから法線ベクトルを計算し，
  // さらに点列の重心より原点からの距離を計算する．
    Vector<double>		eval;
    const Matrix<double>&	Ut = A.eigen(eval);
    _p.resize(d+1);
    _p(0, d) = Ut[Ut.nrow()-1];
    _p[d] = -(_p(0, d)*normalize.centroid());
    if (_p[d] > 0.0)
	_p *= -1.0;
}

//! 与えられた点と超平面の距離の2乗を返す．
/*!
  \param x	点の非同次座標（#spaceDim()次元）または同次座標
		（#spaceDim()+1次元）
  \return	点と超平面の距離の2乗
*/
inline double
HyperPlane::sqdist(const Vector<double>& x) const
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
inline double
HyperPlane::dist(const Vector<double>& x) const
{
    if (x.dim() == spaceDim())
    {
	Vector<double>	xx(spaceDim()+1);
	xx(0, spaceDim()) = x;
	xx[spaceDim()] = 1.0;
	return fabs(_p * xx);
    }
    else if (x.dim() == spaceDim() + 1)
    {
	if (x[spaceDim()] == 0.0)
	    throw std::invalid_argument("HyperPlane::dist(): point at infinitiy!!");
	return fabs((_p * x)/x[spaceDim()]);
    }
    else
	throw std::invalid_argument("HyperPlane::dist(): dimension mismatch!!");

    return 0;
}

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
	virtual Point2<double>
	    operator ()(const Point2<double>& xc)		const	;
	virtual Point2<double>
	    xd(const Point2<double>& xc)			const	;
	virtual Matrix<double>
	    jacobianK(const Point2<double>& xc)			const	;
	virtual Matrix<double>
	    jacobianXC(const Point2<double>& xc)		const	;
	virtual Point2<double>
	    xc(const Point2<double>& u)				const	;

      // calibration matrices.    
	virtual Matrix<double>	K()				const	;
	virtual Matrix<double>	Kt()				const	;
	virtual Matrix<double>	Kinv()				const	;
	virtual Matrix<double>	Ktinv()				const	;

      // intrinsic parameters.
	virtual u_int		dof()				const	;
	virtual double		k()				const	;
	virtual Point2<double>	principal()			const	;
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
    Point2<double>	xc(const Vector<double>& x)		const	;
    Point2<double>	xc(const Point2<double>& u)		const	;
    Matrix<double>	Pc()					const	;
    Matrix<double>	jacobianPc(const Vector<double>& x)	const	;
    Matrix<double>	jacobianXc(const Vector<double>& x)	const	;

  // various oeprations in image coordinates.
    Point2<double>	operator ()(const Vector<double>& x)	const	;
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
    Point2<double>	principal()	const	{return intrinsic().principal();}
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
inline Point2<double>
CameraBase::xc(const Vector<double>& x) const
{
    const Vector<double>&	xx = _Rt * (x - _t);
    return Point2<double>(xx[0] / xx[2], xx[1] / xx[2]);
}

//! 画像座標における点の2次元位置をcanonicalカメラ座標系に直す
/*!
  \param u	画像座標系における点の2次元位置．
  \return	canonicalカメラ座標系におけるuの2次元位置．
*/
inline Point2<double>
CameraBase::xc(const Point2<double>& u) const
{
    return intrinsic().xc(u);
}

//! 3次元空間中の点の像の画像座標系における位置を求める
/*!
  \param x	3次元空間中の点を表す3次元ベクトル．
  \return	xの像の画像座標系における位置．
*/
inline Point2<double>
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
	virtual Point2<double>
	    operator ()(const Point2<double>& xc)		const	;
	virtual Matrix<double>
	    jacobianK(const Point2<double>& xc)			const	;
	virtual Matrix<double>
	    jacobianXC(const Point2<double>& xc)		const	;
	virtual Point2<double>
	    xc(const Point2<double>& u)				const	;

      // calibration matrices.
	virtual Matrix<double>		K()			const	;
	virtual Matrix<double>		Kt()			const	;
	virtual Matrix<double>		Kinv()			const	;
	virtual Matrix<double>		Ktinv()			const	;

      // intrinsic parameters.
	virtual u_int			dof()			const	;
	virtual double			k()			const	;
	virtual	CameraBase::Intrinsic&	setFocalLength(double k)	;

      // parameter updating functions.
	virtual CameraBase::Intrinsic&
	    update(const Vector<double>& dp)				;

      // I/O functions.
	virtual std::istream&		get(std::istream& in)		;
	virtual std::ostream&		put(std::ostream& out)	const	;

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
	virtual Point2<double>
	    operator ()(const Point2<double>& xc)		const	;
	virtual Matrix<double>
	    jacobianK(const Point2<double>& xc)			const	;
	virtual Point2<double>
	    xc(const Point2<double>& u)				const	;
    
      // calibration matrices.	
	virtual Matrix<double>		K()			const	;
	virtual Matrix<double>		Kt()			const	;
	virtual Matrix<double>		Kinv()			const	;
	virtual Matrix<double>		Ktinv()			const	;

      // intrinsic parameters.
	virtual u_int			dof()			const	;
	virtual Point2<double>		principal()		const	;
	virtual CameraBase::Intrinsic&	setPrincipal(double u0,
						     double v0)		;

      // parameter updating functions.
	virtual CameraBase::Intrinsic&  update(const Vector<double>& dp);

      // I/O functions.
	virtual std::istream&		get(std::istream& in)		;
	virtual std::ostream&		put(std::ostream& out)	const	;

      private:
	Point2<double>	_principal;
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
	virtual Point2<double>
	    operator ()(const Point2<double>& xc)		const	;
	virtual Matrix<double>
	    jacobianK(const Point2<double>& xc)			const	;
	virtual Matrix<double>
	    jacobianXC(const Point2<double>& xc)		const	;
	virtual Point2<double>
	    xc(const Point2<double>& u)				const	;

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
	virtual Point2<double>
	    operator ()(const Point2<double>& xc)		const	;
	virtual Point2<double>
	    xd(const Point2<double>& xc)			const	;
	virtual Matrix<double>
	    jacobianXC(const Point2<double>& xc)		const	;
	virtual Matrix<double>
	    jacobianK(const Point2<double>& xc)			const	;
	virtual CameraBase::Intrinsic&
	    update(const Vector<double>& dp)				;
	virtual Point2<double>
	    xc(const Point2<double>& u)				const	;

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
