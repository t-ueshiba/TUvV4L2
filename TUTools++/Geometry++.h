/*
 *  $Id: Geometry++.h,v 1.7 2003-03-17 00:22:30 ueshiba Exp $
 */
#ifndef __TUGeometryPP_h
#define __TUGeometryPP_h

#include "TU/Vector++.h"

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
    Coordinate&		operator =(const CoordinateP<T, D>&)	;

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
*  class CameraBase							*
************************************************************************/
/*!
  すべての透視投影カメラの基底となるクラス．
*/
class CameraBase
{
  public:
  /*!
    カメラの内部パラメータを表すクラス．
  */
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
  //! 位置を原点に，姿勢を単位行列にセットして初期化．
    CameraBase()
	:_t(3), _Rt(3, 3)	{_Rt[0][0] = _Rt[1][1] = _Rt[2][2] = 1.0;}
  //! 位置と姿勢を単位行列にセットして初期化．
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
    virtual Intrinsic&	intrinsic()		= 0;

    Vector<double>	_t;			// camera center.
    Matrix<double>	_Rt;			// camera orientation.
};

//! 3次元空間中の点の像のcanonicalカメラ座標系における位置を求める．
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

//! 画像座標における点の2次元位置をcanonicalカメラ座標系に直す．
/*!
  \param u	画像座標系における点の2次元位置．
  \return	canonicalカメラ座標系におけるuの2次元位置．
*/
inline Point2<double>
CameraBase::xc(const Point2<double>& u) const
{
    return intrinsic().xc(u);
}

//! 3次元空間中の点の像の画像座標系における位置を求める．
/*!
  \param x	3次元空間中の点を表す3次元ベクトル．
  \return	xの像の画像座標系における位置．
*/
inline Point2<double>
CameraBase::operator ()(const Vector<double>& x) const
{
    return intrinsic()(xc(x));
}

//! 3次元ユークリッド空間から画像平面への投影行列を求める．
/*!
  \return	投影行列．
*/
inline Matrix<double>
CameraBase::P() const
{
    return K() * Pc();
}

//! 位置を固定したときの内部/外部パラメータに関するJacobianを求める．
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

  private:
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
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

  private:
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
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

  private:
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
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

  private:
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
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

  private:
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
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
