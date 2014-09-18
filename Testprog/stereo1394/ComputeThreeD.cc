/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2002-2007.
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
 *  $Id: ComputeThreeD.cc 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include "ComputeThreeD.h"

namespace TU
{
/************************************************************************
*  class ComputeThreeD							*
************************************************************************/
ComputeThreeD::ComputeThreeD(const Matrix34d& Pl, const Matrix34d& Pr)
    :_Mt()
{
  // Compute the camera center of Pr.
    Vector4d	tR;
    tR[0] = -Pr[0][3];
    tR[1] = -Pr[1][3];
    tR[2] = -Pr[2][3];
    tR[3] = 1.0;
    tR(0, 3).solve(Pr(0, 0, 3, 3).trns());

  // Compute the transformation matrix.
    Matrix44d	Minv;
    Minv(0, 0, 3, 4) = Pl;
    Minv[3][3]	     = Pl[0] * tR;
    _Mt	= Minv.inv().trns();
}
	
Point3d
ComputeThreeD::operator ()(int u, int v, float d) const
{
    Vector4d	x;
    x[0] = u;
    x[1] = v;
    x[2] = 1.0;
    x[3] = d;
    x *= _Mt;
    
    return x.inhomogeneous();
}

}
