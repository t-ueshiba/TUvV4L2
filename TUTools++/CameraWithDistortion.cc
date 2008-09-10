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
 *  $Id: CameraWithDistortion.cc,v 1.12 2008-09-10 05:10:32 ueshiba Exp $
 */
#include "TU/Camera.h"

namespace TU
{
/************************************************************************
*  class CameraWithDistortion						*
************************************************************************/
CameraBase&
CameraWithDistortion::setProjection(const Matrix34d& PP)
{
    Matrix33d	KK;		// camera intrinsic parameters.
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

    Matrix33d	RRt;		// camera rotation.
    RRt[0]   =  qr.Qt()[2];
    RRt[1]   =  qr.Qt()[1];
    RRt[2]   = -qr.Qt()[0];

    Vector3d	tt;		// camera translation.
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
CameraWithDistortion::intrinsic() const
{
    return _intrinsic;
}

CameraBase::Intrinsic&
CameraWithDistortion::intrinsic()
{
    return _intrinsic;
}

/************************************************************************
*  class CameraWithDistortion::Intrinsic				*
************************************************************************/
Point2d
CameraWithDistortion::Intrinsic::operator ()(const Point2d& xc) const
{
    return Camera::Intrinsic::operator ()(xd(xc));
}

Point2d
CameraWithDistortion::Intrinsic::xd(const Point2d& xc) const
{
    const double	sqr = xc * xc, tmp = 1.0 + sqr*(_d1 + sqr*_d2);
    return Point2d(tmp * xc[0], tmp * xc[1]);
}

Matrix22d
CameraWithDistortion::Intrinsic::jacobianXC(const Point2d& xc) const
{
    const double	sqr = xc * xc, tmp = 2.0*(_d1 + 2.0*sqr*_d2);
    Matrix22d		J;
    J[0][0] = J[1][1] = 1.0 + sqr*(_d1 + sqr*_d2);
    J[0][0] += tmp * xc[0] * xc[0];
    J[1][1] += tmp * xc[1] * xc[1];
    J[0][1] = J[1][0] = tmp * xc[0] * xc[1];
    (J[0] *= k00()) += k01() * J[1];
    J[1] *= k();

    return J;
}

Matrix<double>
CameraWithDistortion::Intrinsic::jacobianK(const Point2d& xc) const
{
    const Point2d&	xxd = xd(xc);
    Matrix<double>	J(2, 7);
    J[1][0] = J[0][4] = xxd[1];
    J[0][1] = J[1][2] = 1.0;
    J[0][3] = xxd[0];
    const double	sqr = xc * xc;
    J[0][5] = sqr * (k00() * xc[0] + k01() * xc[1]);
    J[1][5] = sqr * (		       k() * xc[1]);
    J[0][6] = sqr * J[0][5];
    J[1][6] = sqr * J[1][5];

    return J;
}

Point2d
CameraWithDistortion::Intrinsic::xcFromU(const Point2d& u) const
{
    const Point2d&	xd = Camera::Intrinsic::xcFromU(u);
    const double	sqr = xd * xd, tmp = 1.0 - sqr*(_d1 + sqr*_d2);
    return Point2d(tmp * xd[0], tmp * xd[1]);
}

CameraBase::Intrinsic&
CameraWithDistortion::Intrinsic::update(const Vector<double>& dp)
{
    Camera::Intrinsic::update(dp(0, 5));
    _d1 -= dp[5];
    _d2 -= dp[6];
    return *this;
}

u_int
CameraWithDistortion::Intrinsic::dof() const
{
    return 7;
}

double
CameraWithDistortion::Intrinsic::d1() const
{
    return _d1;
}

double
CameraWithDistortion::Intrinsic::d2() const
{
    return _d2;
}

CameraBase::Intrinsic&
CameraWithDistortion::Intrinsic::setDistortion(double d1, double d2)
{
    _d1 = d1;
    _d2 = d2;
    return *this;
}

std::istream&
CameraWithDistortion::Intrinsic::get(std::istream& in)
{
    Camera::Intrinsic::get(in);
    in >> _d1 >> _d2;

    return in;
}

std::ostream&
CameraWithDistortion::Intrinsic::put(std::ostream& out) const
{
    using namespace	std;
    
    Camera::Intrinsic::put(out);
    cerr << "Distortion-1:    "; out << _d1 << endl;
    cerr << "Distortion-2:    "; out << _d2 << endl;

    return out;
}
 
}
