/*
 *  $Id: CameraBase.cc,v 1.5 2003-02-27 09:09:52 ueshiba Exp $
 */
#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  class CameraBase							*
************************************************************************/
CameraBase::~CameraBase()
{
}

//! 3次元ユークリッド空間からcanonical画像平面への投影行列を求める
/*!
  \return	canonical画像平面への投影行列．
*/
Matrix<double>
CameraBase::Pc() const
{
    Matrix<double>	PP(3, 4);
    PP(0, 0, 3, 3) = _Rt;
    PP[0][3] = -(_Rt[0] * _t);
    PP[1][3] = -(_Rt[1] * _t);
    PP[2][3] = -(_Rt[2] * _t);

    return PP;
}

//! カメラパラメータに関する投影点のcanonical座標の1次微分を求める
/*!
  \param x	対象点の3次元位置を表す3次元ベクトル．
  \return	xにおける投影点のcanonical座標の1次微分．
*/
Matrix<double>
CameraBase::jacobianPc(const Vector<double>& x) const
{
    const Vector<double>&	dx = x - _t;
    const Vector<double>&	xx = _Rt * dx;
    Matrix<double>		J(2, 6);
    (J[0](0, 3) = (xx[0] / xx[2] * _Rt[2] - _Rt[0])) /= xx[2];
    (J[1](0, 3) = (xx[1] / xx[2] * _Rt[2] - _Rt[1])) /= xx[2];
    J[0](3, 3) = J[0](0, 3) ^ dx;
    J[1](3, 3) = J[1](0, 3) ^ dx;

    return J;
}

//! 点の3次元位置に関する投影点のcanonical座標の1次微分を求める
Matrix<double>
CameraBase::jacobianXc(const Vector<double>& x) const
{
    const Vector<double>&	xx = _Rt * (x - _t);
    Matrix<double>		J(2, 3);
    (J[0] = (_Rt[0] - xx[0] / xx[2] * _Rt[2])) /= xx[2];
    (J[1] = (_Rt[1] - xx[1] / xx[2] * _Rt[2])) /= xx[2];

    return J;
}
 
//! カメラパラメータに関する投影点の画像座標の1次微分を求める．
Matrix<double>
CameraBase::jacobianP(const Vector<double>& x) const
{
    const Point2<double>&	xx = xc(x);
    const Matrix<double>&	JK = intrinsic().jacobianK(xx);
    Matrix<double>		J(2, 6 + JK.ncol());
    J(0, 0, 2, 6) = intrinsic().jacobianXC(xx) * jacobianPc(x);
    J(0, 6, 2, JK.ncol()) = JK;
    return J;
}
	
std::istream&
CameraBase::get(std::istream& in)
{
    Vector<double>	axis;
    in >> _t >> axis;
    _Rt = Matrix<double>::Rt(axis);
    return intrinsic().get(in);
}

std::ostream&
CameraBase::put(std::ostream& out) const
{
    std::cerr << "Position:       ";    out << _t;
    std::cerr << "Rotation:       ";    out << _Rt.rot2axis();
    return intrinsic().put(out);
}


/************************************************************************
*  class CameraBase::Intrinsic						*
************************************************************************/
CameraBase::Intrinsic::~Intrinsic()
{
}

Point2<double>
CameraBase::Intrinsic::operator ()(const Point2<double>& xc) const
{
    return xc;
}

Point2<double>
CameraBase::Intrinsic::xd(const Point2<double>& xc) const
{
    return xc;
}

Matrix<double>
CameraBase::Intrinsic::jacobianK(const Point2<double>& xc) const
{
    return Matrix<double>(2, 0);
}

Matrix<double>
CameraBase::Intrinsic::jacobianXC(const Point2<double>& xc) const
{
    return Matrix<double>::I(2);
}
    
Point2<double>
CameraBase::Intrinsic::xc(const Point2<double>& u) const
{
    return u;
}

Matrix<double>
CameraBase::Intrinsic::K() const
{
    return Matrix<double>::I(3);
}
    
Matrix<double>
CameraBase::Intrinsic::Kt() const
{
    return Matrix<double>::I(3);
}
    
Matrix<double>
CameraBase::Intrinsic::Kinv() const
{
    return Matrix<double>::I(3);
}
    
Matrix<double>
CameraBase::Intrinsic::Ktinv() const
{
    return Matrix<double>::I(3);
}

double
CameraBase::Intrinsic::k() const
{
    return 1.0;
}

Point2<double>
CameraBase::Intrinsic::principal() const
{
    return Point2<double>(0.0, 0.0);
}

double
CameraBase::Intrinsic::aspect() const
{
    return 1.0;
}

double
CameraBase::Intrinsic::skew() const
{
    return 0.0;
}

double
CameraBase::Intrinsic::d1() const
{
    return 0.0;
}

double
CameraBase::Intrinsic::d2() const
{
    return 0.0;
}

CameraBase::Intrinsic&
CameraBase::Intrinsic::setFocalLength(double k)
{
    return *this;				// Do nothing.
}

CameraBase::Intrinsic&
CameraBase::Intrinsic::setPrincipal(double u0, double v0)
{
    return *this;				// Do nothing.
}

CameraBase::Intrinsic&
CameraBase::Intrinsic::setAspect(double aspect)
{
    return *this;				// Do nothing.
}

CameraBase::Intrinsic&
CameraBase::Intrinsic::setSkew(double skew)
{
    return *this;				// Do nothing.
}

CameraBase::Intrinsic&
CameraBase::Intrinsic::setIntrinsic(const Matrix<double>& K)
{
    return *this;				// Do nothing.
}

CameraBase::Intrinsic&
CameraBase::Intrinsic::setDistortion(double d1, double d2)
{
    return *this;				// Do nothing.
}
	
CameraBase::Intrinsic&
CameraBase::Intrinsic::update(const Vector<double>& dp)
{
    return *this;				// Do nothing.
}

std::istream&
CameraBase::Intrinsic::get(std::istream& in)
{
    return in;					// Do nothing.
}

std::ostream&
CameraBase::Intrinsic::put(std::ostream& out) const
{
    return out;					// Do nothing.
}
 
}
