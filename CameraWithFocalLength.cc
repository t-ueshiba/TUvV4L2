/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: CameraWithFocalLength.cc,v 1.9 2008-09-08 08:06:13 ueshiba Exp $
 */
#include "TU/Camera.h"

namespace TU
{
/************************************************************************
*  class CameraWithFocalLength						*
************************************************************************/
CameraBase&
CameraWithFocalLength::setProjection(const Matrix34d& PP)
{
    Matrix33d			Nt = PP(0, 0, 3, 3);
    Nt[0] *= 0.5;
    Nt[1] *= 0.5;
    SVDecomposition<double>	svd(Nt);
    if (svd[2] < 0.0)
	throw std::invalid_argument("CameraWithFocalLength::setProjection: cannot extract camera rotation due to nevative singular value!");
    const Matrix<double>&	RRt = svd.Vt().trns() * svd.Ut();
    const double		p = RRt[0]*Nt[0] + RRt[1]*Nt[1],
				q = RRt[2]*Nt[2];
    Vector3d			tc;
    tc[0] = -PP[0][3] / p;
    tc[1] = -PP[1][3] / p;
    tc[2] = -PP[2][3] / q;
    setFocalLength(p / q).setRotation(RRt).setTranslation(tc * RRt);

    return *this;
}

const CameraBase::Intrinsic&
CameraWithFocalLength::intrinsic() const
{
    return _intrinsic;
}

/*
 *  private members
 */
CameraBase::Intrinsic&
CameraWithFocalLength::intrinsic()
{
    return _intrinsic;
}

/************************************************************************
*  class CameraWithFocalLength::Intrinsic				*
************************************************************************/
Point2d
CameraWithFocalLength::Intrinsic::operator ()(const Point2d& xc) const
{
    return Point2d(_k * xc[0], _k * xc[1]);
}

Matrix<double>
CameraWithFocalLength::Intrinsic::jacobianK(const Point2d& xc) const
{
    Matrix<double>	J(2, 1);
    J[0][0] = xc[0];
    J[1][0] = xc[1];

    return J;
}

Matrix22d
CameraWithFocalLength::Intrinsic::jacobianXC(const Point2d& xc) const
{
    Matrix22d	J;
    return J.diag(_k);
}
    
Point2d
CameraWithFocalLength::Intrinsic::xcFromU(const Point2d& u) const
{
    return Point2d(u[0] / _k, u[1] / _k);
}

Matrix33d
CameraWithFocalLength::Intrinsic::K() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = _k;
    mat[2][2] = 1.0;

    return mat;
}
    
Matrix33d
CameraWithFocalLength::Intrinsic::Kt() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = _k;
    mat[2][2] = 1.0;

    return mat;
}
    
Matrix33d
CameraWithFocalLength::Intrinsic::Kinv() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = 1.0 / _k;
    mat[2][2] = 1.0;

    return mat;
}
    
Matrix33d
CameraWithFocalLength::Intrinsic::Ktinv() const
{
    Matrix33d	mat;
    mat[0][0] = mat[1][1] = 1.0 / _k;
    mat[2][2] = 1.0;

    return mat;
}

u_int
CameraWithFocalLength::Intrinsic::dof() const
{
    return 1;
}

double
CameraWithFocalLength::Intrinsic::k() const
{
    return _k;
}

CameraBase::Intrinsic&
CameraWithFocalLength::Intrinsic::setFocalLength(double k)
{
    _k = k;
    return *this;
}    

CameraBase::Intrinsic&
CameraWithFocalLength::Intrinsic::update(const Vector<double>& dp)
{
    _k -= dp[0];
    return *this;
}

std::istream&
CameraWithFocalLength::Intrinsic::get(std::istream& in)
{
    CanonicalCamera::Intrinsic::get(in);
    return in >> _k;
}

std::ostream&
CameraWithFocalLength::Intrinsic::put(std::ostream& out) const
{
    using namespace	std;
    
    CanonicalCamera::Intrinsic::put(out);
    cerr << "Focal length:    ";
    return out << _k << endl;
}
 
}
