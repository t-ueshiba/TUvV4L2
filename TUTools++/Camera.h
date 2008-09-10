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
 *  $Id: Camera.h,v 1.2 2008-09-10 05:10:32 ueshiba Exp $
 */
#ifndef __TUCamera_h
#define __TUCamera_h

#include "TU/Geometry++.h"

namespace TU
{
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
	virtual Matrix22d	jacobianXC(const Point2d& xc)	const	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;

      // calibration matrices.    
	virtual Matrix33d	K()				const	;
	virtual Matrix33d	Kt()				const	;
	virtual Matrix33d	Kinv()				const	;
	virtual Matrix33d	Ktinv()				const	;

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
	virtual Intrinsic&	setIntrinsic(const Matrix33d& K)	;
	virtual Intrinsic&	setDistortion(double d1, double d2)	;

      // parameter updating functions.
	virtual Intrinsic&	update(const Vector<double>& dp)	;

      // I/O functions.
	virtual std::istream&	get(std::istream& in)			;
	virtual std::ostream&	put(std::ostream& out)		const	;
    };
    
  public:
  //! 位置を原点に，姿勢を単位行列にセットして初期化
    CameraBase() :_t(), _Rt()	{_Rt[0][0] = _Rt[1][1] = _Rt[2][2] = 1.0;}
  //! 位置と姿勢を単位行列にセットして初期化
  /*!
    \param t	カメラ位置を表す3次元ベクトル．
    \param Rt	カメラ姿勢を表す3x3回転行列．
  */
    CameraBase(const Point3d& t, const Matrix33d& Rt)	:_t(t), _Rt(Rt)	{}
    virtual ~CameraBase()						;
    
  // various operations in canonical coordinates.
    Point2d		xc(const Point3d& x)			const	;
    Point2d		xcFromU(const Point2d& u)		const	;
    Matrix34d		Pc()					const	;
    Matrix<double>	jacobianPc(const Point3d& x)		const	;
    Matrix23d		jacobianXc(const Point3d& x)		const	;

  // various oeprations in image coordinates.
    Point2d		operator ()(const Point3d& x)		const	;
    Matrix34d		P()					const	;
    Matrix<double>	jacobianP(const Point3d& x)		const	;
    Matrix<double>	jacobianFCC(const Point3d& x)		const	;
    Matrix23d		jacobianX(const Point3d& x)		const	;
    Matrix<double>	jacobianK(const Point3d& x)		const	;
    Matrix22d		jacobianXC(const Point3d& x)		const	;
    virtual CameraBase& setProjection(const Matrix34d& P)		=0;

  // parameter updating functions.
    void		update(const Vector<double>& dp)		;
    void		updateFCC(const Vector<double>& dp)		;
    void		updateIntrinsic(const Vector<double>& dp)	;
    
  // calibration matrices.
    Matrix33d		K()		const	{return intrinsic().K();}
    Matrix33d		Kt()		const	{return intrinsic().Kt();}
    Matrix33d		Kinv()		const	{return intrinsic().Kinv();}
    Matrix33d		Ktinv()		const	{return intrinsic().Ktinv();}

  // extrinsic parameters.
    const Point3d&	t()		const	{return _t;}
    const Matrix33d&	Rt()		const	{return _Rt;}
    CameraBase&		setTranslation(const Point3d& t)	;
    CameraBase&		setRotation(const Matrix33d& Rt)	;

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
    CameraBase&		setIntrinsic(const Matrix33d& K)	;
    CameraBase&		setDistortion(double d1, double d2)	;
    
  // I/O functions.
    std::istream&	get(std::istream& in)			;
    std::ostream&	put(std::ostream& out)		const	;

  private:
    Point3d		_t;			// camera center.
    Matrix33d		_Rt;			// camera orientation.
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
CameraBase::xc(const Point3d& x) const
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
CameraBase::xcFromU(const Point2d& u) const
{
    return intrinsic().xcFromU(u);
}

//! 3次元空間中の点の像の画像座標系における位置を求める
/*!
  \param x	3次元空間中の点を表す3次元ベクトル．
  \return	xの像の画像座標系における位置．
*/
inline Point2d
CameraBase::operator ()(const Point3d& x) const
{
    return intrinsic()(xc(x));
}

//! 3次元ユークリッド空間から画像平面への投影行列を求める
/*!
  \return	投影行列．
*/
inline Matrix34d
CameraBase::P() const
{
    return K() * Pc();
}

//! 位置を固定したときの内部/外部パラメータに関するJacobianを求める
/*!
  \return	
*/
inline Matrix<double>
CameraBase::jacobianFCC(const Point3d& x) const
{
    const Matrix<double>&	J = jacobianP(x);
    return Matrix<double>(J, 0, 3, J.nrow(), J.ncol() - 3);
}

inline Matrix23d
CameraBase::jacobianX(const Point3d& x) const
{
    return intrinsic().jacobianXC(xc(x)) * jacobianXc(x);
}

inline Matrix<double>
CameraBase::jacobianK(const Point3d& x) const
{
    return intrinsic().jacobianK(xc(x));
}

inline Matrix22d
CameraBase::jacobianXC(const Point3d& x) const
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
CameraBase::setTranslation(const Point3d& t)
{
    _t = t;
    return *this;
}

inline CameraBase&
CameraBase::setRotation(const Matrix33d& Rt)
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
CameraBase::setIntrinsic(const Matrix33d& K)
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
    CanonicalCamera(const Point3d& t, const Matrix33d& Rt)
	:CameraBase(t, Rt), _intrinsic()		{}
    CanonicalCamera(const Matrix34d& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}
    
    virtual CameraBase&	setProjection(const Matrix34d& P)		;
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
	virtual Matrix22d	jacobianXC(const Point2d& xc)	const	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;

      // calibration matrices.
	virtual Matrix33d	K()				const	;
	virtual Matrix33d	Kt()				const	;
	virtual Matrix33d	Kinv()				const	;
	virtual Matrix33d	Ktinv()				const	;

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
    CameraWithFocalLength(const Point3d& t,
			  const Matrix33d& Rt, double k=1.0)
	:CameraBase(t, Rt), _intrinsic(k)			{}
    CameraWithFocalLength(const Matrix34d& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&		setProjection(const Matrix34d& P)	;
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
	virtual Point2d		xcFromU(const Point2d& u)	const	;
    
      // calibration matrices.	
	virtual Matrix33d	K()				const	;
	virtual Matrix33d	Kt()				const	;
	virtual Matrix33d	Kinv()				const	;
	virtual Matrix33d	Ktinv()				const	;

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
    CameraWithEuclideanImagePlane(const Point3d&	t,
				  const Matrix33d&	Rt,
				  double		k=1.0,
				  double		u0=0,
				  double		v0=0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0)			{}
    CameraWithEuclideanImagePlane(const Matrix34d& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&	setProjection(const Matrix34d& P)		;
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
	Intrinsic(const Matrix33d& K)
	    :CameraWithEuclideanImagePlane::Intrinsic(),
	     _k00(k()), _k01(0.0)			{setIntrinsic(K);}
	
      // various operations.
	virtual Point2d		operator ()(const Point2d& xc)	const	;
	virtual Matrix<double>	jacobianK(const Point2d& xc)	const	;
	virtual Matrix22d	jacobianXC(const Point2d& xc)	const	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;

      // calibration matrices.
	virtual Matrix33d	K()				const	;
	virtual Matrix33d	Kt()				const	;
	virtual Matrix33d	Kinv()				const	;
	virtual Matrix33d	Ktinv()				const	;

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
				setIntrinsic(const Matrix33d& K)	;

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
    Camera(const Point3d&	t,
	   const Matrix33d&	Rt,
	   double		k=1.0,
	   double		u0=0,
	   double		v0=0,
	   double		aspect=1.0,
	   double		skew=0.0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0, aspect, skew)	{}
    Camera(const Matrix34d& P)
	:CameraBase(), _intrinsic()			{setProjection(P);}

    virtual CameraBase&	setProjection(const Matrix34d& P)		;
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
	Intrinsic(const Matrix33d& K)
	    :Camera::Intrinsic(), _d1(0.0), _d2(0.0)	{setIntrinsic(K);}
	
      // various operations.
	virtual Point2d		operator ()(const Point2d& xc)	const	;
	virtual Point2d		xd(const Point2d& xc)		const	;
	virtual Matrix<double>	jacobianK(const Point2d& xc)	const	;
	virtual Matrix22d	jacobianXC(const Point2d& xc)	const	;
	virtual CameraBase::Intrinsic&
				update(const Vector<double>& dp)	;
	virtual Point2d		xcFromU(const Point2d& u)	const	;

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
    CameraWithDistortion(const Point3d&		t,
			 const Matrix33d&	Rt,
			 double			k=1.0,
			 double			u0=0,
			 double			v0=0,
			 double			aspect=1.0,
			 double			skew=0.0,
			 double			d1=0.0,
			 double			d2=0.0)
	:CameraBase(t, Rt), _intrinsic(k, u0, v0, aspect, skew, d1, d2)	{}
    CameraWithDistortion(const Matrix34d& P,
			 double d1=0.0, double d2=0.0)			;

    virtual CameraBase&		setProjection(const Matrix34d& P)	;
    virtual const CameraBase::Intrinsic&	intrinsic()	const	;
    virtual CameraBase::Intrinsic&		intrinsic()		;
    
  private:
    Intrinsic	_intrinsic;
};
 
inline
CameraWithDistortion::CameraWithDistortion(const Matrix34d& P,
					   double d1, double d2)
    :CameraBase(), _intrinsic()
{
    setProjection(P);
    setDistortion(d1, d2);
}

}
#endif	/* !__TUCamera_h */
