/*
 *  $Id: CameraWithDistortion.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include "TU/Geometry++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class CameraWithDistortion						*
************************************************************************/
CameraBase&
CameraWithDistortion::setProjection(const Matrix<double>& PP)
{
    throw std::runtime_error("CameraWithDistortion::setProjection: Not implemented!!");
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
Point2<double>
CameraWithDistortion::Intrinsic
		      ::operator ()(const Point2<double>& xc) const
{
    return Camera::Intrinsic::operator ()(xd(xc));
}

Matrix<double>
CameraWithDistortion::Intrinsic
		      ::jacobianXC(const Point2<double>& xc) const
{
    const double	sqr = xc * xc, tmp = 2.0*(_d1 + 2.0*sqr*_d2);
    Matrix<double>	J(2, 2);
    J[0][0] = J[1][1] = 1.0 + sqr*(_d1 + sqr*_d2);
    J[0][0] += tmp * xc[0] * xc[0];
    J[1][1] += tmp * xc[1] * xc[1];
    J[0][1] = J[1][0] = tmp * xc[0] * xc[1];
    (J[0] *= k00()) += k01() * J[1];
    J[1] *= k();

    return J;
}

Matrix<double>
CameraWithDistortion::Intrinsic::jacobianK(const Point2<double>& xc) const
{
    const Point2<double>	xxd = xd(xc);
    Matrix<double>		J(2, 7);
    J[1][0] = J[0][4] = xxd[1];
    J[0][1] = J[1][2] = 1.0;
    J[0][3] = xxd[0];
    const double		sqr = xc * xc;
    J[0][5] = sqr * (k00() * xc[0] + k01() * xc[1]);
    J[1][5] = sqr * (		       k() * xc[1]);
    J[0][6] = sqr * J[0][5];
    J[1][6] = sqr * J[1][5];

    return J;
}

CameraBase::Intrinsic&
CameraWithDistortion::Intrinsic::update(const Vector<double>& dp)
{
    Camera::Intrinsic::update(dp(0, 5));
    _d1 -= dp[5];
    _d2 -= dp[6];
    return *this;
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
    cerr << "Distortion-2     "; out << _d2 << endl;

    return out;
}
 
}
