/*
 *  平成19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  同所が著作権を所有する秘密情報です．著作者による許可なしにこのプロ
 *  グラムを第三者へ開示，複製，改変，使用する等の著作権を侵害する行為
 *  を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *  Copyright 2007
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Author: Toshio UESHIBA
 *
 *  Confidentail and all rights reserved.
 *  This program is confidential. Any changing, copying or giving
 *  information about the source code of any part of this software
 *  and/or documents without permission by the authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damages in the use of this program.
 *  
 *  $Id: Camera.cc,v 1.6 2007-11-26 07:28:09 ueshiba Exp $
 */
#include "TU/Geometry++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class Camera								*
************************************************************************/
CameraBase&
Camera::setProjection(const Matrix<double>& PP)
{
    if (PP.nrow() != 3 || PP.ncol() != 4)
	throw std::invalid_argument("Camera::setProjection: Illegal dimension of P!!");

    Matrix<double>	KK(3, 3);	// camera intrinsic parameters.
    KK[0]    = PP[2](0, 3);
    KK[1]    = PP[1](0, 3);
    KK[2]    = PP[0](0, 3);
    QRDecomposition<double>	qr(KK);
    KK[0][0] =  qr.Rt()[2][2];
    KK[0][1] =  qr.Rt()[2][1];
    KK[0][2] = -qr.Rt()[2][0];
    KK[1][0] =  0.0;
    KK[1][1] =  qr.Rt()[1][1];
    KK[1][2] = -qr.Rt()[1][0];
    KK[2][0] =  0.0;
    KK[2][1] =  0.0;
    KK[2][2] = -qr.Rt()[0][0];

    Matrix<double>	RRt(3, 3);	// camera rotation.
    RRt[0]   =  qr.Qt()[2];
    RRt[1]   =  qr.Qt()[1];
    RRt[2]   = -qr.Qt()[0];

    Vector<double>	tt(3);		// camera translation.
    tt[0]    = -PP[0][3];
    tt[1]    = -PP[1][3];
    tt[2]    = -PP[2][3];

  // Negate sign of PP so that KK has positive determinant.
    if (KK[0][0] * KK[1][1] * KK[2][2] < 0.0)
    {
	KK *= -1.0;
	tt *= -1.0;
    }
    
    if (KK[0][0] < 0.0)
    {
	KK[0][0] *= -1.0;
	RRt[0] *= -1.0;
    }
    if (KK[1][1] < 0.0)
    {
	KK[0][1] *= -1.0;
	KK[1][1] *= -1.0;
	RRt[1] *= -1.0;
    }
    if (KK[2][2] < 0.0)
    {
	KK[0][2] *= -1.0;
	KK[1][2] *= -1.0;
	KK[2][2] *= -1.0;
	RRt[2] *= -1.0;
    }
    tt = (KK.inv() * tt) * RRt;

    setIntrinsic(KK).setTranslation(tt).setRotation(RRt);

    return *this;
}

/*
 *  private members
 */
const CameraBase::Intrinsic&
Camera::intrinsic() const
{
    return _intrinsic;
}

CameraBase::Intrinsic&
Camera::intrinsic()
{
    return _intrinsic;
}

/************************************************************************
*  class Camera::Intrinsic						*
************************************************************************/
Point2<double>
Camera::Intrinsic::operator ()(const Point2<double>& xc) const
{
    return Point2<double>(_k00 * xc[0] + _k01 * xc[1] + principal()[0],
					    k() * xc[1] + principal()[1]);
}

Matrix<double>
Camera::Intrinsic::jacobianK(const Point2<double>& xc) const
{
    Matrix<double>	J(2, 5);
    J[1][0] = J[0][4] = xc[1];
    J[0][1] = J[1][2] = 1.0;
    J[0][3] = xc[0];

    return J;
}

Matrix<double>
Camera::Intrinsic::jacobianXC(const Point2<double>& xc) const
{
    Matrix<double>	J(2, 2);
    J[0][0] = _k00;
    J[0][1] = _k01;
    J[1][1] = k();

    return J;
}

Point2<double>
Camera::Intrinsic::xc(const Point2<double>& u) const
{
    return Point2<double>((u[0] - principal()[0] -
			   (u[1] - principal()[1]) * _k01 / k()) / _k00,
			  (u[1] - principal()[1]) / k());
}

Matrix<double>
Camera::Intrinsic::K() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = _k00;
    mat[0][1] = _k01;
    mat[0][2] = principal()[0];
    mat[1][1] = k();
    mat[1][2] = principal()[1];
    mat[2][2] = 1.0;

    return mat;
}

Matrix<double>
Camera::Intrinsic::Kt() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = _k00;
    mat[1][0] = _k01;
    mat[2][0] = principal()[0];
    mat[1][1] = k();
    mat[2][1] = principal()[1];
    mat[2][2] = 1.0;

    return mat;
}

Matrix<double>
Camera::Intrinsic::Kinv() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = 1.0 / _k00;
    mat[0][1] = -_k01 / (_k00 * k());
    mat[0][2] = -principal()[0] * mat[0][0] - principal()[1] * mat[0][1];
    mat[1][1] = 1.0 / k();
    mat[1][2] = -principal()[1] / k();
    mat[2][2] = 1.0;

    return mat;
}

Matrix<double>
Camera::Intrinsic::Ktinv() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = 1.0 / _k00;
    mat[1][0] = -_k01 / (_k00 * k());
    mat[2][0] = -principal()[0] * mat[0][0] - principal()[1] * mat[1][0];
    mat[1][1] = 1.0 / k();
    mat[2][1] = -principal()[1] / k();
    mat[2][2] = 1.0;

    return mat;
}

u_int
Camera::Intrinsic::dof() const
{
    return 5;
}

double
Camera::Intrinsic::aspect() const
{
    return _k00 / k();
}

double
Camera::Intrinsic::skew() const
{
    return _k01 / k();
}

CameraBase::Intrinsic&
Camera::Intrinsic::setFocalLength(double kk)
{
    _k00 *= (kk / k());
    _k01 *= (kk / k());
    return CameraWithFocalLength::Intrinsic::setFocalLength(kk);
}

CameraBase::Intrinsic&
Camera::Intrinsic::setIntrinsic(const Matrix<double>& K)
{
    setAspect(K[0][0] / K[1][1])
	.setSkew(K[0][1] / K[1][1])
	.setPrincipal(K[0][2]/K[2][2], K[1][2]/K[2][2])
	.setFocalLength(K[1][1]/K[2][2]);

    return *this;
}

CameraBase::Intrinsic&
Camera::Intrinsic::setAspect(double aspect)
{
    _k00 = aspect * k();
    return *this;
}

CameraBase::Intrinsic&
Camera::Intrinsic::setSkew(double skew)
{
    _k01 = skew * k();
    return *this;
}

CameraBase::Intrinsic&
Camera::Intrinsic::update(const Vector<double>& dp)
{
    CameraWithEuclideanImagePlane::Intrinsic::update(dp(0, 3));
    _k00 -= dp[3];
    _k01 -= dp[4];
    return *this;
}

std::istream&
Camera::Intrinsic::get(std::istream& in)
{
    CameraWithEuclideanImagePlane::Intrinsic::get(in);
    in >> _k00 >> _k01;
    _k00 *= k();
    _k01 *= k();

    return in;
}

std::ostream&
Camera::Intrinsic::put(std::ostream& out) const
{
    using namespace	std;
    
    CameraWithEuclideanImagePlane::Intrinsic::put(out);
    cerr << "Aspect ratio:    "; out << aspect() << endl;
    cerr << "Skew:            "; out << skew() << endl;

    return out;
}
 
}
