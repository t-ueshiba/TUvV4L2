/*
 *  $Id: CanonicalCamera.cc,v 1.3 2003-02-27 09:09:52 ueshiba Exp $
 */
#include "TU/Geometry++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class CanonicalCamera						*
************************************************************************/
CameraBase&
CanonicalCamera::setProjection(const Matrix<double>& PP)
{
    SVDecomposition<double>	svd(PP(0, 0, 3, 3));
    if (svd[2] < 0.0)
	throw std::runtime_error("TU::CanonicalCamera::setProjection: cannot extract camera rotation due to nevative singular value!");
    setRotation(svd.Vt().trns() * svd.Ut());
    Vector<double>	tc(3);
    tc[0] = -PP[0][3];
    tc[1] = -PP[1][3];
    tc[2] = -PP[2][3];
    setTranslation(tc * Rt());

    return *this;
}

const CameraBase::Intrinsic&
CanonicalCamera::intrinsic() const
{
    return _intrinsic;
}

/*
 *  private members
 */
CameraBase::Intrinsic&
CanonicalCamera::intrinsic()
{
    return _intrinsic;
}
 
}
