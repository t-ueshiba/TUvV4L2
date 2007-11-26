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
 *  $Id: CanonicalCamera.cc,v 1.4 2007-11-26 07:28:09 ueshiba Exp $
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
