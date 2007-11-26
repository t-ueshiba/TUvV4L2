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
 *  $Id: CameraWithEuclideanImagePlane.cc,v 1.8 2007-11-26 07:55:48 ueshiba Exp $
 */
#include "TU/Geometry++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class CameraWithEuclideanImagePlane					*
************************************************************************/
CameraBase&
CameraWithEuclideanImagePlane::setProjection(const Matrix<double>& PP)
{
    throw std::runtime_error("CameraWithEuclideanImagePlane::setProjection: Not implemented!!");
    return *this;
}

/*
 *  private members
 */
const CameraBase::Intrinsic&
CameraWithEuclideanImagePlane::intrinsic() const
{
    return _intrinsic;
}

CameraBase::Intrinsic&
CameraWithEuclideanImagePlane::intrinsic()
{
    return _intrinsic;
}

/************************************************************************
*  class CameraWithEuclideanImagePlane::Intrinsic			*
************************************************************************/
Point2<double>
CameraWithEuclideanImagePlane
    ::Intrinsic::operator ()(const Point2<double>& xc) const
{
    return Point2<double>(k() * xc[0] + _principal[0],
			  k() * xc[1] + _principal[1]);
}

Matrix<double>
CameraWithEuclideanImagePlane
    ::Intrinsic::jacobianK(const Point2<double>& xc) const
{
    Matrix<double>	J(2, 3);
    J[0][0] = xc[0];
    J[1][0] = xc[1];
    J[0][1] = J[1][2] = 1.0;

    return J;
}

Point2<double>
CameraWithEuclideanImagePlane::Intrinsic::xc(const Point2<double>& u) const
{
    return Point2<double>((u[0] - _principal[0]) / k(),
			  (u[1] - _principal[1]) / k());
}

Matrix<double>
CameraWithEuclideanImagePlane::Intrinsic::K() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = mat[1][1] = k();
    mat[0][2] = _principal[0];
    mat[1][2] = _principal[1];
    mat[2][2] = 1.0;

    return mat;
}

Matrix<double>
CameraWithEuclideanImagePlane::Intrinsic::Kt() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = mat[1][1] = k();
    mat[2][0] = _principal[0];
    mat[2][1] = _principal[1];
    mat[2][2] = 1.0;

    return mat;
}

Matrix<double>
CameraWithEuclideanImagePlane::Intrinsic::Kinv() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = mat[1][1] = 1.0 / k();
    mat[0][2] = -_principal[0] / k();
    mat[1][2] = -_principal[1] / k();
    mat[2][2] = 1.0;

    return mat;
}

Matrix<double>
CameraWithEuclideanImagePlane::Intrinsic::Ktinv() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = mat[1][1] = 1.0 / k();
    mat[2][0] = -_principal[0] / k();
    mat[2][1] = -_principal[1] / k();
    mat[2][2] = 1.0;

    return mat;
}

u_int
CameraWithEuclideanImagePlane::Intrinsic::dof() const
{
    return 3;
}

Point2<double>
CameraWithEuclideanImagePlane::Intrinsic::principal() const
{
    return _principal;
}

CameraBase::Intrinsic&
CameraWithEuclideanImagePlane::Intrinsic::setPrincipal(double u0, double v0)
{
    _principal[0] = u0;
    _principal[1] = v0;
    return *this;
}

CameraBase::Intrinsic&
CameraWithEuclideanImagePlane::Intrinsic::update(const Vector<double>& dp)
{
    CameraWithFocalLength::Intrinsic::update(dp(0, 1));
    _principal[0] -= dp[1];
    _principal[1] -= dp[2];
    return *this;
}

std::istream&
CameraWithEuclideanImagePlane::Intrinsic::get(std::istream& in)
{
    CameraWithFocalLength::Intrinsic::get(in);
    return in >> _principal;
}

std::ostream&
CameraWithEuclideanImagePlane::Intrinsic::put(std::ostream& out) const
{
    CameraWithFocalLength::Intrinsic::put(out);
    std::cerr << "Principal point:";
    return out << _principal << std::endl;
}
 
}
