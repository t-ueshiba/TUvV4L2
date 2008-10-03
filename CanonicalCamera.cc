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
 *  $Id: CanonicalCamera.cc,v 1.9 2008-10-03 04:23:37 ueshiba Exp $
 */
#include "TU/Camera.h"

namespace TU
{
/************************************************************************
*  class CanonicalCamera						*
************************************************************************/
CameraBase&
CanonicalCamera::setProjection(const Matrix34d& PP)
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

CameraBase::Intrinsic&
CanonicalCamera::intrinsic()
{
    return _intrinsic;
}
    
}
