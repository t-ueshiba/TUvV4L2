/*
 *  $Id: CameraWithFocalLength.cc,v 1.3 2002-10-28 00:37:01 ueshiba Exp $
 */
#include "TU/Geometry++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class CameraWithFocalLength					*
************************************************************************/
CameraBase&
CameraWithFocalLength::setProjection(const Matrix<double>& PP)
{
    Matrix<double>	Nt = PP(0, 0, 3, 3);
    Nt[0] *= 0.5;
    Nt[1] *= 0.5;
    SVDecomposition<double>	svd(Nt);
    if (svd[2] < 0.0)
	throw std::invalid_argument("CameraWithFocalLength::setProjection: cannot extract camera rotation due to nevative singular value!");
    const Matrix<double>&	RRt = svd.Vt().trns() * svd.Ut();
    const double		p = RRt[0]*Nt[0] + RRt[1]*Nt[1], q = RRt[2]*Nt[2];
    Vector<double>	tc(3);
    tc[0] = -PP[0][3] / p;
    tc[1] = -PP[1][3] / p;
    tc[2] = -PP[2][3] / q;
    setFocalLength(p / q).setRotation(RRt).setTranslation(tc * RRt);

    return *this;
}

/*
 *  private members
 */
const CameraBase::Intrinsic&
CameraWithFocalLength::intrinsic() const
{
    return _intrinsic;
}

CameraBase::Intrinsic&
CameraWithFocalLength::intrinsic()
{
    return _intrinsic;
}

/************************************************************************
*  class CameraWithFocalLength::Intrinsic				*
************************************************************************/
Point2<double>
CameraWithFocalLength::Intrinsic
		       ::operator ()(const Point2<double>& xc) const
{
    return Point2<double>(_k * xc[0], _k * xc[1]);
}

Matrix<double>
CameraWithFocalLength::Intrinsic::jacobianK(const Point2<double>& xc) const
{
    Matrix<double>	J(2, 1);
    J[0][0] = xc[0];
    J[1][0] = xc[1];

    return J;
}

Matrix<double>
CameraWithFocalLength::Intrinsic::jacobianXC(const Point2<double>& xc) const
{
    Matrix<double>	J(2, 2);
    return J.diag(_k);
}
    
Point2<double>
CameraWithFocalLength::Intrinsic::xc(const Point2<double>& u) const
{
    return Point2<double>(u[0] / _k, u[1] / _k);
}

Matrix<double>
CameraWithFocalLength::Intrinsic::K() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = mat[1][1] = _k;
    mat[2][2] = 1.0;

    return mat;
}
    
Matrix<double>
CameraWithFocalLength::Intrinsic::Kt() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = mat[1][1] = _k;
    mat[2][2] = 1.0;

    return mat;
}
    
Matrix<double>
CameraWithFocalLength::Intrinsic::Kinv() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = mat[1][1] = 1.0 / _k;
    mat[2][2] = 1.0;

    return mat;
}
    
Matrix<double>
CameraWithFocalLength::Intrinsic::Ktinv() const
{
    Matrix<double>	mat(3, 3);
    mat[0][0] = mat[1][1] = 1.0 / _k;
    mat[2][2] = 1.0;

    return mat;
}

double
CameraWithFocalLength::Intrinsic::k() const
{
    return _k;
}

CameraWithFocalLength::Intrinsic&
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
